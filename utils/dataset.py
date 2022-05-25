import os
import json

import torch 
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 
import matplotlib.image as image 
import numpy as np

import pandas as pd
import albumentations as A
import albumentations.pytorch
import cv2
import math

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch.nn as nn
import torch.nn.functional as F

# IMAGE_SIZE = 448
# IMAGE_SIZE = 224
# box more than this size, crop it
# if backbone is swin, size should be selected carefully
MAX_IMAGE_SIZE = 2048

# adaptive resizing threshold for small image
# if area( geometric mean of W*H is smaller than this, resize in some range (1, 1.5)
SMALL_IMAGE_SIZE = 100.
range_limit = 0.5 # range(1, 1 + range limit)

train_transforms = A.Compose([
    # A.RandomScale(scale_limit=.05, p=0.7),
    A.Resize(MAX_IMAGE_SIZE, MAX_IMAGE_SIZE, p=1),
    A.OneOf([
        A.HorizontalFlip(p=.8),
        A.VerticalFlip(p=.8),
        A.RandomRotate90(p=.8)]
    ),
    A.OneOf([
        A.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=.8),
        A.ToGray(p=0.2),
        A.transforms.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.8)
        ]
    )
], p=1.0,  bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.8, 
                                    label_fields=['labels_one', 'labels_five', 'labels_hpv'])) 

val_transforms = A.Compose([
    A.Resize(MAX_IMAGE_SIZE, MAX_IMAGE_SIZE, p=1),
    A.HorizontalFlip(p=.001),
], p=1.0,  bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.8, 
                                    label_fields=['labels_one', 'labels_five', 'labels_hpv'])) 

test_transforms = A.Compose([
    A.Resize(MAX_IMAGE_SIZE, MAX_IMAGE_SIZE, p=1),
    A.HorizontalFlip(p=.001),
], p=1.0,  bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.8, 
                                    label_fields=['labels_one', 'labels_five', 'labels_hpv']))

def label_mapper(label) :
    label = str(label)
    if label == 'AS' or 'ASC-US with HPV infection' in label:
        return 'ASC-US'
    elif label == 'AH' or 'ASC-H with HPV infection' in label:
        return 'ASC-H' 
    elif label == 'LS' or 'LSIL with HPV infection' in label:
        return 'LSIL'
    elif label == 'HS' or label == 'HN' or 'HSIL with HPV infection' in label :
        return 'HSIL'
    elif label == 'SM' or label == 'SC' :
        return 'Carcinoma'
    elif label == 'C' or label == 'T' or label == 'H' or label == 'AC' :
        return 'Benign'
    else :
        return label

def label_id(label) :
    label = str(label)
    if label == 'ASC-US' :
        return 0
    elif label == 'LSIL' :
        return 1
    elif label == 'HSIL' :
        return 2
    elif label == 'ASC-H' :
        return 3
    elif label == 'Carcinoma' :
        return 4
    elif label == 'Negative' :
        return 5
    else : #others
        return 6

class PapsDetDataset(Dataset):
    def __init__(self, df, defaultpath='../lbp_data/', transform=None):
        self.df = df
        self.num_classes = 1 # one class detection, Abnormal
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])
        print(self.df.shape)
        
        self.transform = transform
        self.dir = defaultpath
        
        # sorting by size, batch should be consider size for compute efficiency and performance
        # I think batchnorm will be affected if image with lots of padding, therefore
        # custom sampler should check area
        self.df = self.df.sort_values('area', axis=0)
        self.df.reset_index(inplace=True, drop=False)

        # abnormal or background
        self.df.label_det_one = self.df.label_det_one.apply(lambda x : int(1) )
        
        # bedesda 5 class except for Negative
        self.df.label_det = self.df.label_det.apply(lambda x : label_id(x))        
        
        # HPV or not HPV or background
        self.df['label_hpv'] = self.df.label.apply(lambda x : int(0) if 'HPV' in x else int(1) )

    def __len__(self):
        # return len(self.df)   
        return len(set(self.df.ID.values))
    
    def __getitem__(self, idx):
        index = self.df[self.df['ID'] == idx].index.values
        label_one = self.df.loc[index].label_det_one.values
        label_five = self.df.loc[index].label_det.values
        label_hpv = self.df.loc[index].label_hpv.values
        area = self.df.loc[index].area.values
        bbox = self.df.loc[index][['xmin', 'ymin', 'w', 'h']].values

        path = self.df.loc[index].file_name.values[0]
        image = cv2.imread(self.dir + path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            timage = self.transform(image=image, bboxes=list(bbox), 
                                    labels_one=label_one,
                                    labels_five=label_five,
                                    labels_hpv=label_hpv
                                   )
            image = timage['image']
            bbox = np.array(timage['bboxes'])
            label_one = timage['labels_one']
            label_five = timage['labels_five']
            label_hpv = timage['labels_hpv']
        
        # retinanet use xmin, ymin, xmax, ymax
        if len(bbox) > 0 :
            bbox[:,2] += bbox[:,0]
            bbox[:,3] += bbox[:,1]
            
        image = image/255.
        image = (image - self.image_mean[None, None, :]) / self.image_std[None, None:, ]
        # image = image.permute(2,0,1)        
        image = np.transpose(image, (2,0,1))
        image = torch.tensor(image, dtype=torch.float32)

        iscrowd = torch.zeros((len(index)), dtype=torch.int64)
        target = {}
        target['boxes'] = torch.as_tensor(bbox, dtype=torch.float32)
        # target['category_id'] = torch.as_tensor(label, dtype=torch.long) 
        target['labels'] = torch.as_tensor(label_one, dtype=torch.long) 
        target['labels_five'] = torch.as_tensor(label_five, dtype=torch.long) 
        target['labels_hpv'] = torch.as_tensor(label_hpv, dtype=torch.long) 
        target["image_id"] = torch.as_tensor(idx, dtype=torch.long)
        target["area"] = torch.as_tensor(area , dtype=torch.float32) 
        target["iscrowd"] = iscrowd   
        
        return image, target
    
