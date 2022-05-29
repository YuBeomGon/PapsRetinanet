import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler, SequentialSampler, RandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter


def get_weight_random_sampler(dataset) :
    
    count = Counter(dataset.df.label_cls)
    
    # 6 class including Negative
    class_count = np.array([count[0],count[1], count[2],count[3], count[4],count[5]])
    
    # use square root (fittign to small samples)
    weight = np.sqrt(1./class_count)
    
    samples_weight = np.array([weight[t] for t in dataset.df.label_cls.values])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))    
    
    return sampler