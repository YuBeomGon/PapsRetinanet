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
from PIL import Image

MAX_IMAGE_SIZE = 2048
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
        A.transforms.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.8)
        ]
    )
], p=1.0,  bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.8, label_fields=['labels'])) 

val_transforms = A.Compose([
    A.Resize(MAX_IMAGE_SIZE, MAX_IMAGE_SIZE, p=1),
    # A.HorizontalFlip(p=.001),
], p=1.0,  bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.8, label_fields=['labels'])) 

test_transforms = A.Compose([
    A.Resize(MAX_IMAGE_SIZE, MAX_IMAGE_SIZE, p=1),
    # A.HorizontalFlip(p=.001),
], p=1.0,  bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.8, label_fields=['labels']))

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

        self.df.label_det_one = self.df.label_det_one.apply(lambda x : int(1) )
        
        # retinanet use xmin, ymin, xmax, ymax
        self.df['xmax'] = self.df.apply(lambda x : x['xmin'] + x['w'], axis=1)
        self.df['ymax'] = self.df.apply(lambda x : x['ymin'] + x['h'], axis=1)        

    def __len__(self):
        # return len(self.df)   
        return len(set(self.df.ID.values))
    
    def __getitem__(self, idx):
        index = self.df[self.df['ID'] == idx].index.values
        label = self.df.loc[index].label_det_one.values
        area = self.df.loc[index].area.values
        bbox = self.df.loc[index][['xmin', 'ymin', 'xmax', 'ymax']].values

        path = self.df.loc[index].file_name.values[0]
        image = cv2.imread(self.dir + path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            timage = self.transform(image=image, bboxes=list(bbox), labels=label)
            image = timage['image']
            bbox = np.array(timage['bboxes'])
            label = timage['labels']
            
        image = image/255.
        image = (image - self.image_mean[None, None, :]) / self.image_std[None, None, :]
        # image = image.permute(2,0,1)        
        image = np.transpose(image, (2,0,1))
        image = torch.tensor(image, dtype=torch.float32)

        iscrowd = torch.zeros((len(index)), dtype=torch.int64)
        target = {}
        target['boxes'] = torch.as_tensor(bbox, dtype=torch.float32)
        target['category_id'] = torch.as_tensor(label, dtype=torch.long) 
        target['labels'] = torch.as_tensor(label, dtype=torch.long) 
        target["image_id"] = torch.as_tensor(idx, dtype=torch.long)
        target["area"] = torch.as_tensor(area , dtype=torch.float32) 
        target["iscrowd"] = iscrowd   
        
        return image, target
    
class PapsClsDataset(Dataset):
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

        self.df.label_cls = self.df.label_cls.apply(lambda x : label_id(x))
        
        # retinanet use xmin, ymin, xmax, ymax
        self.df['xmax'] = self.df.apply(lambda x : x['xmin'] + x['w'], axis=1)
        self.df['ymax'] = self.df.apply(lambda x : x['ymin'] + x['h'], axis=1)

    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        label = self.df.loc[idx]['label_cls']
        area = self.df.loc[idx]['area']
        bbox = self.df.loc[idx][['xmin', 'ymin', 'xmax', 'ymax']].values

        path = self.df.loc[idx]['file_name']
        image = cv2.imread(self.dir + path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # it takes more time than cv2
        # image = Image.open(self.dir + path)
        # image = image.convert("RGB")
        # image = np.array(image)
        
        if self.transform:
            timage = self.transform(image=image, bboxes=[bbox], labels=[label])
            image = timage['image']
            bbox = np.array(timage['bboxes'])
            label = timage['labels']
            
        if len(bbox) == 0 :
            return None
        
        return image, bbox, label    
    
