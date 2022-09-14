import pathlib
import os
from typing import *
from monai.transforms import *

import numpy as np
import pandas as pd
import cv2

def get_filename(filename):
    path = pathlib.Path(filename)
    return path.stem




labels_mapping = {255: 0,
                0: 1,
                80: 2,
                160: 3}


def read_metadata(dataset_folder:str, read_labels:bool=True):
    """
    Read metadata in form of dict for each scan

    :param dataset_folder:
    :param read_labels:
    :return:
    """
    image_folder = os.path.join(dataset_folder,'Image')
    preliminary_folder = os.path.join(dataset_folder,'Preliminary')
    ids = [f.split('.')[0] for f in os.listdir(image_folder) if f.endswith('.png')]
    ids.sort()

    if read_labels:
        segmentations_folder = os.path.join(dataset_folder,'Image')
        #glaucoma_labels = pd.read_excel(os.path.join(dataset_folder,'Train_GC_GT.xlsx'),dtype={'ImgName':str,
        #                                                                                       'GC_Label':np.float32})
        #glaucoma_labels['ImgName'] = glaucoma_labels['ImgName'].str.zfill(4)
        #glaucoma_labels = glaucoma_labels.set_index('ImgName')

        #assert(sorted(ids)==sorted(glaucoma_labels.index))

    for id in ids:
        record = {'id':id,
                  'image':os.path.join(dataset_folder,'Image',id+'.png'),
                    'preliminary': cv2.resize(np.loadtxt(os.path.join(preliminary_folder, id+"_positions.csv"), delimiter=',', dtype=np.float32), (1100,5), interpolation=cv2.INTER_CUBIC).astype(np.int32),
                 }

        if read_labels:
            record['segmentation'] = os.path.join(dataset_folder,'Layer_Masks',id+'.png')
        #    record['glaucoma'] = glaucoma_labels.loc[id,'GC_Label']

        yield record

def read_metadata_crop(dataset_folder:str, read_labels:bool=True, crop_count:int=5):
    """
    Read metadata in form of dict for each scan

    :param dataset_folder:
    :param read_labels:
    :return:
    """
    image_folder = os.path.join(dataset_folder,'Image')
    preliminary_folder = os.path.join(dataset_folder,'Preliminary')
    ids = [f.split('.')[0] for f in os.listdir(image_folder) if f.endswith('.png')]
    ids.sort()

    if read_labels:
        segmentations_folder = os.path.join(dataset_folder,'Image')
        glaucoma_labels = pd.read_excel(os.path.join(dataset_folder,'Train_GC_GT.xlsx'),dtype={'ImgName':str,
                                                                                               'GC_Label':np.float32})
        glaucoma_labels['ImgName'] = glaucoma_labels['ImgName'].str.zfill(4)
        glaucoma_labels = glaucoma_labels.set_index('ImgName')

        assert(sorted(ids)==sorted(glaucoma_labels.index))

    for id in ids:
        #Ugly as fuck
        for crop_id in range(crop_count):
            record = {'id':id,
                    'image':os.path.join(dataset_folder,'Image',id+'.png'),
                    'preliminary': cv2.resize(np.loadtxt(os.path.join(preliminary_folder, id+"_positions.csv"), delimiter=',', dtype=np.float32), (1100,5), interpolation=cv2.INTER_CUBIC).astype(np.int32),
                    'crop_id': crop_id
                    }

            if read_labels:
                record['segmentation'] = os.path.join(dataset_folder,'Layer_Masks',id+'.png')
                record['glaucoma'] = glaucoma_labels.loc[id,'GC_Label']

            yield record

def filter_metadata(metadata:List[dict], ids:List[str]):
    """
    select cases listed in ids from the metadata

    :param metadata:
    :param ids:
    :return:
    """
    return [r for r in metadata if r['id'] in ids]

def get_train_validation_samples(split_id):
    dataset_splits = pd.read_csv('../../splits/goals.v1.csv',index_col=0)
    dataset_splits['id'] = dataset_splits['id'].astype(str).str.zfill(4)

    training_ids, validation_ids = \
        dataset_splits[dataset_splits['split_id'] != split_id]['id'], dataset_splits[dataset_splits['split_id'] == split_id]['id']
    return training_ids, validation_ids

class ConvertToMultiChannelGOALS(MapTransform):
    """
    Convert labels of ong image into 0,1,2,3
    0 - background
    1 - RNFL
    2 - GCIPL
    3 - Choroid
    """

    def __call__(self, data):


        d = dict(data)
        for key in self.keys:

            mask = data[key]

            mask_new = np.zeros_like(mask,dtype=np.uint8)

            for k,v in labels_mapping.items():
                mask_new[mask==k] = v

            d[key] = mask_new
        return d



'''
metadata_generator = read_metadata(dataset_folder='/media/optima/exchange/Dmitrii/GOALS/Train',
                                      read_labels=True)
metadata = list(metadata_generator)
pass
'''