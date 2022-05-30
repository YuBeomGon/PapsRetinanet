import os
import timm
import logging
import argparse
import pandas as pd
from typing import Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchmetrics import Accuracy, F1Score, Specificity

from pytorch_lightning import LightningModule
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import Trainer
# from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from utils.dataset import PapsDetDataset, train_transforms, val_transforms, test_transforms, MAX_IMAGE_SIZE
from utils.collate import det_collate_fn
#from utils.sampler_by_group import GroupedBatchSampler, create_area_groups
#from utils.losses import SupConLoss, FocalLoss

from models.detection.backbone import PapsBackboneWithFPN
# from vision.torchvision.models.detection.retinanet import RetinaNet
from models.detection.retinanet import PapsRetinaNet
from det_utils.engine import train_one_epoch, evaluate

# import custom_models
# from models.efficientnet import EfficientNet, VALID_MODELS

parser = argparse.ArgumentParser(description='PyTorch Lightning ImageNet Training')
parser.add_argument('--data_path', metavar='DIR', default='./lbp_data/',
                    help='path to dataset (default: ./lbp_data/)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    help='model architecture: (default: resnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--accelerator', '--accelerator', default='gpu', type=str, help='default: gpu')

parser.add_argument('--devices', '--devices', default=4, type=int, help='number of gpus, default 2')
parser.add_argument('--img_size', default=400, type=int, help='input image resolution in swin models')
parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
# parser.add_argument('--groups', default=3, type=int, help='number of groups of data')
# parser.add_argument('--drop_last', default=False, type=bool, help='drop or not on every end of epoch or groups')

parser.add_argument('--pretrained', default=False, type=bool, help='set True if using pretrained weights')
parser.add_argument('--output_dir', default='./saved_models/detection', type=str, help='directory for model checkpoint')


class PapsDetModel(LightningLite) :

    def init (self, args) :
        self.data_path = args.data_path
        self.arch = args.arch
        self.pretrained = args.pretrained
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.num_classes = args.num_classes
        self.out_channels = 256
        self.epochs = args.epochs
        self.arch = args.arch
        self.output_dir = args.output_dir

        self.backbone = PapsBackboneWithFPN(self.arch, self.out_channels)
        
        # including background class
        self.model = PapsRetinaNet(self.backbone, num_classes=self.num_classes + 1, anchor_generator=None)

        # self.save_hyperparameters()
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms   
        self.print_freq = 100

    def SetDataLoader(self,) :

        self.train_df = pd.read_csv(self.data_path + '/train_det.csv') 
        self.train_dataset = PapsDetDataset(self.train_df, defaultpath=self.data_path, transform=self.train_transforms)        

        self.test_df = pd.read_csv(self.data_path + '/test_det.csv')
        self.eval_dataset = PapsDetDataset(self.test_df, defaultpath=self.data_path, transform=self.test_transforms)        

        train_loader = torch.utils.data.DataLoader(
                                                        dataset=self.train_dataset,
                                                        batch_size=self.batch_size, # batch_size is decided in sampler
                                                        shuffle=True,
                                                        num_workers=self.workers,
                                                        collate_fn=det_collate_fn,
                                                        # pin_memory=True,
                                                        # drop_last=True
                                                        )
        self.train_loader = self.setup_dataloaders(train_loader)

        test_loader = torch.utils.data.DataLoader(
                                                        dataset=self.eval_dataset,
                                                        batch_size=self.batch_size, # batch_size is decided in sampler
                                                        shuffle=False,
                                                        num_workers=self.workers,
                                                        collate_fn=det_collate_fn,
                                                        # pin_memory=True,
                                                        # drop_last=True
                                                        )
        self.test_loader = self.setup_dataloaders(test_loader)

    def run(self, args) :

        self.init(args)
        self.SetDataLoader()
        self.fit()
        
        
    def fit(self) :
        optimizer, scheduler = self.configure_optimizers()
        self.model, optimizer = self.setup(self.model, optimizer)         
        
        for epoch in range(1, self.epochs) :
            self.model.train()
            train_one_epoch(self, optimizer, epoch, scheduler, scaler=None)
            # scheduler.step()

            # if epoch > 20 and epoch % 5 == 0 :
            evaluate(self.model, self.test_loader)
            
            if self.output_dir:
                self.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_{epoch}.pth"))
                self.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.pth"))
            
    def configure_optimizers(self) :
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        epochs              = self.epochs, 
                                                        steps_per_epoch     = len(self.train_loader),
                                                        max_lr              = self.lr, 
                                                        pct_start           = 0.1, 
                                                        div_factor          = 100, 
                                                        final_div_factor    = 2e+4)   
        
        # scheduler = {'scheduler': scheduler, 'interval': 'step'}
        
        return optimizer, scheduler

if __name__ == '__main__' :
    args = parser.parse_args() 
    trainer_defaults = dict(
        # plugins = "deepspeed_stage_2_offload",
        precision = 16,
        accelerator = 'gpu', # auto, or select device, "gpu"
        devices = 1, # number of gpus
        # devices = 1, # number of gpus
        # logger = [logger_tb, logger_wandb],
        strategy = "ddp",
        # gpus=[0],
        ) 
    
    # PapsDetModel(accelerator='gpu', devices=-1).run(args)
    PapsDetModel(**trainer_defaults).run(args)


