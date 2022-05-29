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

from utils.dataset import PapsClsDataset, train_transforms, val_transforms, test_transforms, MAX_IMAGE_SIZE
from utils.collate import collate_fn
# from utils.sampler_by_group import GroupedBatchSampler, create_area_groups
from utils.losses import SupConLoss, FocalLoss

# from cls_utils.block import Bottleneck, TwoMLPHead, RoIPool
from cls_utils.model import PapsClassificationModel
from utils.collate import collate_fn
from utils.sampler import get_weight_random_sampler


# import custom_models
# from models.efficientnet import EfficientNet, VALID_MODELS

parser = argparse.ArgumentParser(description='PyTorch Lightning ImageNet Training')
parser.add_argument('--data_path', metavar='DIR', default='./lbp_data/',
                    help='path to dataset (default: ./lbp_data/)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: (default: resnet18)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--accelerator', '--accelerator', default='gpu', type=str, help='default: gpu')

parser.add_argument('--devices', '--devices', default=4, type=int, help='number of gpus, default 2')
parser.add_argument('--img_size', default=400, type=int, help='input image resolution in swin models')
parser.add_argument('--num_classes', default=6, type=int, help='number of classes')

parser.add_argument('--pretrained', default=True, type=bool, help='set True if using pretrained weights')
parser.add_argument('--output_dir', default='./saved_models/classification', type=str, help='directory for model checkpoint')


class PapsClsModel(LightningModule) :
    def __init__(
        self,
        data_path : str,
        arch: str = 'resnet18',
        pretrained: bool = False,
        lr: float = 0.9,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int =256,
        workers: int = 16,
        num_classes: int = 5,
    ):
        
        super().__init__()
        self.data_path = data_path
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.workers = workers
        self.num_classes = num_classes
        
        self.epochs = args.epochs

        # get just top feature map only
        self.model = PapsClassificationModel(self.arch, self.pretrained, self.num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = FocalLoss()
            
        print("=> creating model '{}'".format(self.arch))
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.train_acc1 = Accuracy(top_k=1)
        self.eval_acc1 = Accuracy(top_k=1)
        self.f1 = F1Score(average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(average='macro', num_classes=self.num_classes)
        
        self.save_hyperparameters()
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        
    def forward(self, x, boxes) :
        x = self.model(x, boxes)
        return x
    
    def training_step(self, batch, batch_idx) :
        images, boxes, labels = batch
        labels = labels.squeeze()
        outputs = self(images, boxes)
        
        loss = self.criterion(outputs, labels)
        correct=outputs.argmax(dim=1).eq(labels).sum().item()
        total=len(labels)
        
        #update metric
        self.log('train_loss', loss)
        self.train_acc1(outputs, labels)
        self.log('train_acc', self.train_acc1, prog_bar=True)
        
        #for tensorboard
        logs={"train_loss": loss}
        batch_dictionary={
            'loss':loss,
            'log':logs,
            # info to be used at epoch end
            "correct": correct,
            "total": total
        }        
        
        return batch_dictionary
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])
        
        # creating log dictionary
        tensorboard_logs = {'loss': avg_loss, "Accuracy": correct/total}
        
        epoch_dictionary={
            # required
            'loss': avg_loss,
            # for logging purposes
            'log': tensorboard_logs
        }
 
        # wandb expect None
        # return epoch_dictionary  

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
    
    def eval_step(self, batch, batch_idx, prefix: str) :
        images, boxes, labels = batch
        labels = labels.squeeze()
        outputs = self(images, boxes)
        
        loss = self.criterion(outputs, labels)
        
        self.log(f'{prefix}_loss', loss)
        self.eval_acc1(outputs, labels)
        self.log(f'{prefix}_acc1', self.eval_acc1, prog_bar=True)
        self.f1(outputs, labels)
        self.log(f'{prefix}_f1_score', self.f1, prog_bar=True)
        self.specificity(outputs, labels)
        self.log(f'{prefix}_specificity', self.specificity, prog_bar=True)    
        
        if prefix == 'val' :
            correct=outputs.argmax(dim=1).eq(labels).sum().item()
            total=len(labels) 
            
            #for tensorboard
            logs={"val_loss": loss}
            batch_dictionary={
                'loss':loss,
                'log':logs,
                # info to be used at epoch end
                "correct": correct,
                "total": total
            }  
            
            return batch_dictionary

        return loss            
        
    def validation_step(self, batch, batch_idx) :
        return self.eval_step(batch, batch_idx, 'val')
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])
        
        tensorboard_logs = {'loss': avg_loss, "Accuracy": correct/total}
        
        epoch_dictionary={
            # required
            'loss': avg_loss,
            # for logging purposes
            'log': tensorboard_logs
        }
        
        # wandb expect None
        # return epoch_dictionary    

    def test_step(self, batch, batch_idx) :
        return self.eval_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self) :
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), 
        #                       lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        
        # scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch : 0.1 **(epoch //30))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        epochs              = self.epochs, 
                                                        steps_per_epoch     = int(len(self.train_dataset)/self.batch_size),
                                                        max_lr              = self.lr, 
                                                        pct_start           = 0.1, 
                                                        div_factor          = 100, 
                                                        final_div_factor    = 2e+4)   
        
        scheduler = {'scheduler': scheduler, 'interval': 'step'}        
        return [optimizer], [scheduler]
    
    def setup(self, stage: Optional[str] = None) :
        if isinstance(self.trainer.strategy, ParallelStrategy) :
            # When using a single GPU per process and per `DistributedDataParallel`, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)   
            
        if stage in (None, 'fit') :
            train_df = pd.read_csv(self.data_path + '/train.csv') 
            self.train_dataset = PapsClsDataset(train_df, defaultpath=self.data_path, transform=self.train_transforms)
            
        #train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)  
        # self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        self.train_sampler = get_weight_random_sampler(self.train_dataset)
        
        test_df = pd.read_csv(self.data_path + '/test.csv')
        self.eval_dataset = PapsClsDataset(test_df, defaultpath=self.data_path, transform=self.test_transforms)         
        # test_sampler = torch.utils.data.distributed.DistributedSampler(self.eval_dataset)

        
    def train_dataloader(self) :
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size, # batch_size is decided in sampler
            shuffle=False,
            sampler=self.train_sampler,
            collate_fn=collate_fn,
            num_workers=self.workers,
            # pin_memory=True,
            # drop_last=True
        )
    
    def val_dataloader(self) :
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.workers,
            # pin_memory=True
        )
    
    def test_dataloader(self) :
        return self.val_dataloader()
    
#     load contra checkpoint in fine tuning
    def load_contra_checkpoint(self, path):
        state_dict = torch.load(path)['state_dict']
        model_state_dict = self.state_dict()
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]

            else:
                print(f"Dropping parameter {k}")

        self.load_state_dict(state_dict)
        
#     model freezing when fine tuning( linear evaluation protocol)
    def model_freeze(self) :
        for param in self.parameters( ) :
            param.requires_grad = False
        
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True


if __name__ == "__main__":
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    args = parser.parse_args()
    if torch.cuda.is_available() :
        args.accelerator = 'gpu'
        args.devices = torch.cuda.device_count()
        
    args.img_size = MAX_IMAGE_SIZE
    # tb_logger = TensorBoardLogger(save_dir="tuning_logs/" + args.arch, name=args.arch + "_my_model")
    logger_tb = TensorBoardLogger('./tuning_logs' +'/' + args.arch, name=now)
    logger_wandb = WandbLogger(project='Paps_clf', name=now, mode='online') # online or disabled    
    
    trainer_defaults = dict(
        callbacks = [
            # the PyTorch example refreshes every 10 batches
            TQDMProgressBar(refresh_rate=50),
            # save when the validation top1 accuracy improves
            ModelCheckpoint(monitor="val_acc1", mode="max",
                            dirpath=args.output_dir + '/' + args.arch,
                            filename='paps_tunning_{epoch}_{val_acc1:.2f}'),  
            ModelCheckpoint(monitor="val_acc1", mode="max",
                            dirpath=args.output_dir + '/' + args.arch,
                            filename='paps_tunning_best'),             
        ],    
        # plugins = "deepspeed_stage_2_offload",
        precision = 16,
        max_epochs = args.epochs,
        accelerator = args.accelerator, # auto, or select device, "gpu"
        devices = args.devices, # number of gpus
        # devices = 1, # number of gpus
        logger = [logger_tb, logger_wandb],
        benchmark = True,
        strategy = "ddp",
        replace_sampler_ddp=False,
        # gpus=[1],
        )
    
    model = PapsClsModel(
        data_path=args.data_path,
        arch=args.arch,
        pretrained=args.pretrained,
        workers=args.workers,
        lr = args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
    )
    
    # path = detection path
    '''
    if os.path.isdir(path) and 'paps-contra_best.ckpt' in os.listdir(path) :
        print('checkpoint is loaded from ', path)
        model.load_contra_checkpoint(path + '/paps-contra_best.ckpt')   
#         model freeze except last fcn layer
        print('model freeze except last fc layer')
        model.model_freeze()
    '''
     
    trainer = Trainer(**trainer_defaults)
    trainer.fit(model)  
    
    trainer.test(model)


