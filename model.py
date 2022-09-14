import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import architecture.sdnet
from config import Config
from utils.gdice import GeneralizedDiceLoss
from architecture.common import DifferentiableRounding
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import img
from optimizers import ranger
from optima.io import csvLayer

from utils import utils
import monai
import monai.data
import pathlib


class SDNet(pl.LightningModule):

    def __init__(self,
                 config : Config):
        super().__init__()
        self.config = config
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.sdnet = architecture.sdnet.SDNet(img_size = config.img_size,
                                              n_encoder_latent=config.n_encoder_latent,
                                              n_classes=config.n_classes,
                                              n_anatomical_factors=config.n_anatomical_factors,
                                              use_segmentor=False, layer_engine = config.layer_engine, drop_rate = config.drop_rate)
        self.dice_loss = GeneralizedDiceLoss()
        
        self.num_workers = self.config.num_workers
        self.batch_size = self.config.batch_size
        
        self.target_size = (config.img_size, config.img_size)
        
        self.mseloss = nn.MSELoss()
        self.kld = nn.KLDivLoss()
        self.relu = nn.ReLU()
        self.metrics = []
        self.training_samples, self.validation_samples = utils.get_train_validation_samples(self.config.split_id)
    

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loader_sup = DataLoader(self.load_dataset(self.config.train_sup_dataset_path), batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True, persistent_workers=True, drop_last=True)
                                  
        loader_unsup = DataLoader(self.load_unsup_dataset(self.config.train_unsup_dataset_path),
                                  batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True, persistent_workers=True, drop_last=True)
        
        loaders = {"sup": loader_sup, "unsup": loader_unsup}
        
        return loaders
    
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.load_dataset_val(self.config.validation_path), batch_size=2,
                                  num_workers=self.num_workers, pin_memory=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.load_test_dataset("/path/to/dataset"),
                                  batch_size=20, shuffle=False,
                                  num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
        
        
    def training_step(self, batch, batch_idx):
        result = {}
        supervised_loss = self.training_step_supervised(batch["sup"], batch_idx)
        alpha = 0.5
        unsupervised_loss = self.training_step_unsupervised(batch["unsup"], batch_idx)
        
        result["loss"] = alpha * supervised_loss + (1 - alpha) * unsupervised_loss 
        
        return result
        # --------------------------
    
    def training_step_supervised(self, batch, batch_idx):
        prob_map, layer_positions, clean_masks, hard_anatomy, extra_losses = self.sdnet.get_layer_anatomical_factors(batch["image"])
        z_mean, z_logvar, sampled_z = self.sdnet.get_modalities(batch["image"], hard_anatomy)
        reconstructed_img = self.sdnet.get_reconstructed_img(hard_anatomy, sampled_z)
        
        kl_i = 1.0 + z_logvar - (z_mean ** 2) - torch.exp(z_logvar)
        kl_div_loss = -0.5 * torch.sum(kl_i, dim=1)
        kl_div_loss = torch.mean(kl_div_loss)
        
        recon_filter = (hard_anatomy[:, 0:1, ...] + hard_anatomy[:, (self.config.n_classes-1):(self.config.n_classes), ...]) < 0.5
        if recon_filter.sum() > 1:
            rec_loss = torch.mean(torch.abs(reconstructed_img[recon_filter] - batch["image"][recon_filter]))
        else:
            rec_loss = torch.mean(torch.abs(reconstructed_img - batch["image"]))
        
        z_estimate = self.sdnet.get_z_estimate(reconstructed_img.detach(), hard_anatomy.detach())
        
        
        z_mean_detached = z_mean.detach()
        z_regress_loss = torch.mean(torch.abs(z_estimate - z_mean_detached))
        w_kl = 0.1
        w_segm = 50.0
        w_rec = 1.0
        w_z_rec = 1.0
        w_xentr = 0.10
        
        mp =  batch["mask_positions"][:, :, 0, :]
        
        segmentation_kld = self.kld(prob_map, batch["mask_probability_map"])
        segmentation_mse = self.mseloss(layer_positions, mp / self.config.img_size)
        
        topology_violations = 0.1 * self.relu(extra_losses["topology_violations"]).sum() / self.config.img_size
        continuity_violations = 0.1 * self.relu(extra_losses["continuity_violations"] - 1.5).sum()/ self.config.img_size
        diff_10_violations = self.relu(extra_losses["diff_10"] - 0.5).sum() / self.config.img_size
        curvature_diffs = self.relu(extra_losses["curvature_diffs"]).mean()
        
        filter = np.array([0, 1, 1, 0, 1, 0], dtype=np.bool)
        filtered_masks = clean_masks[:, filter, ...]
        segmentation_loss_area = self.dice_loss(filtered_masks, batch["masks"])
        segmentation_loss =  segmentation_kld + segmentation_mse + 0.5 * segmentation_loss_area
        
        
        loss = w_kl * kl_div_loss + w_segm * segmentation_loss + w_rec * rec_loss \
                + w_z_rec * z_regress_loss + topology_violations + continuity_violations + diff_10_violations + curvature_diffs
        if torch.isnan(loss) or torch.isinf(loss):
            print("training_step_supervised")
        
        return loss
    
    def training_step_unsupervised(self, batch, batch_idx):
        _, _, _, hard_anatomy, extra_losses = self.sdnet.get_layer_anatomical_factors(batch["image"])
        z_mean, z_logvar, sampled_z = self.sdnet.get_modalities(batch["image"], hard_anatomy)
        reconstructed_img = self.sdnet.get_reconstructed_img(hard_anatomy, sampled_z)
        
        kl_i = 1.0 + z_logvar - (z_mean ** 2) - torch.exp(z_logvar)
        kl_div_loss = -0.5 * torch.sum(kl_i, dim=1)
        kl_div_loss = torch.mean(kl_div_loss)
        
        recon_filter = (hard_anatomy[:, 0:1, ...] + hard_anatomy[:, (self.config.n_classes-1):(self.config.n_classes), ...]) < 0.5
        if recon_filter.sum() > 1:
            rec_loss = torch.mean(torch.abs(reconstructed_img[recon_filter] - batch["image"][recon_filter]))
        else:
            rec_loss = torch.mean(torch.abs(reconstructed_img - batch["image"]))
        
        z_estimate = self.sdnet.get_z_estimate(reconstructed_img.detach(), hard_anatomy.detach())
        
        z_mean_detached = z_mean.detach()
        z_regress_loss = torch.mean(torch.abs(z_estimate - z_mean_detached))
        
        
        w_kl = 0.1
        w_segm = 10.0
        w_rec = 1.0
        w_z_rec = 1.0
        w_adv = 1.0
        w_xentr = 0.10
        
        std_deviations = 0.1 * self.relu(extra_losses["std_deviations"] - 1).mean()
        
        curvature_diffs = self.relu(extra_losses["curvature_diffs"]).mean()
        topology_violations = 0.1 * self.relu(extra_losses["topology_violations"]).sum() / self.config.img_size
        continuity_violations = 0.1 * self.relu(extra_losses["continuity_violations"] - 1.5).sum() / self.config.img_size
        
        
        loss = w_kl * kl_div_loss + w_rec * rec_loss \
                + w_z_rec * z_regress_loss + std_deviations + topology_violations + continuity_violations + curvature_diffs
                        
        
        return loss
        


    def configure_optimizers(self):
        sup_optimizer = ranger.Ranger(self.sdnet.parameters(), lr=self.config.learning_rate)
        
        return sup_optimizer
