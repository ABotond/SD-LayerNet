import pathlib
import json
import os

class Config:
    
    def __init__(self) -> None:
        self.learning_rate = 0.001
        self.batch_size = 7
        self.n_anatomical_factors = 12
        self.n_classes = 12
        self.img_size = 256
        self.n_encoder_latent = 15
        self.layer_engine = 1
        self.validation_path = ("/path")
        self.train_sup_dataset_path = ("/path")
        self.train_unsup_dataset_path = ("/path")
        self.num_workers = 5
        self.metrics_dir = ""
        self.drop_rate = 0.0
        self.flip_rate = 0.0
        self.split_id = 0
        self.n_epoch = 50
        self.um_conversion_rate = 6.05
        self.extra_layer_penalty = 0.0001
 
    @classmethod
    def load_from_json(cls, filename : pathlib.Path):
        with open(filename) as json_file:
            data = json.load(json_file)
            config = Config()
            config.__dict__ = data
            config.experiment_name = filename.parent.name + "_" + os.path.splitext(filename.name)[0]
            config.split_name = os.path.splitext(filename.name)[0]
            return config
    
    def save_config(self, filename):
        with open(filename, "w") as json_file:
            json.dump(self.__dict__, json_file, indent=4)
