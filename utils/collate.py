import numpy as np
import torch

image_mean = np.array([0.485, 0.456, 0.406])
image_std = np.array([0.229, 0.224, 0.225])

def collate_fn(batch):

    images = np.stack([b[0] for b in batch if b != None])
    images = images/255.
    images = (images - image_mean[None, None, None, :]) / image_std[None, None, None, :]
    images = torch.tensor(np.transpose(images, (0,3,1,2)), dtype=torch.float32)
    
    boxes = np.stack([b[1] for b in batch if b != None])
    boxes = torch.tensor(boxes, dtype=torch.float32)
    
    labels = torch.tensor([b[2] for b in batch if b != None], dtype=torch.long)
    
    return images, boxes, labels