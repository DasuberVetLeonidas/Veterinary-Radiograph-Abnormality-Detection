from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision.transforms.functional import pad as PAD
import torch.optim as optim
from torch.utils.data.dataset import random_split
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
import re
import pandas as pd
from sklearn.metrics import classification_report
import wandb
import numbers
os.environ['CUDA_VISIBLE_DEVICES']='0'
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import ToPILImage,ToTensor
ToPILImage = ToPILImage()

hyperparameter_defaults = dict(
    input_size = 1024,
    batch_size = 5,
    training_depth = 19,
    num_epochs = 49,
    learning_rate = 0.0006007,
    betas1 = 0.7807,
    betas2 = 0.905,
    eps = 0.000001976,
    amsgrad = False)

wandb.init(project="Finetune With MURA Dataset",config=hyperparameter_defaults)
wbconfig = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ### Helper Functions

def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return PAD(img, get_padding(img), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


def test(model, dataloaders, device):
    labels = []
    preds = []
    total = 0
    num_correct = 0
    total_imgs = len(dataloaders['test'].dataset)
    imgs_done = 0
    for _,param in model.named_parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    running_accuracy = 0
    for inputs,label in dataloaders['test']:
        imgs_done+=1
        progress = (imgs_done/total_imgs*100)
        
        wandb.log({"Testing Progress":progress})
        inputs = inputs.to(device)
        total += 1
        outputs = model(inputs)
        _,pred = torch.max(outputs,1)
        pred = pred.to('cpu').detach()  
        labels.append(int(label.detach()))
        preds.append(int(pred))
        if label == pred:
            num_correct += 1
            running_accuracy = (num_correct/imgs_done)*100
        print("Current testing progress\t{:.2f}% running accuracy:\t{:.4f}%".format(progress,running_accuracy),end='\r')
        wandb.log({"Running Accuracy":round(running_accuracy,4)})
    return labels,preds

data_transforms = {
    'train': transforms.Compose([
        NewPad(),
        transforms.RandomRotation(180),
        transforms.Resize(wbconfig.input_size),
        transforms.ColorJitter(0.2,0.2,0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.1523, 0.1523, 0.1523], [0.1402, 0.1402, 0.1402])
    ]),
    'val': transforms.Compose([
        NewPad(),
        transforms.Resize(wbconfig.input_size),
        transforms.CenterCrop(wbconfig.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.1523, 0.1523, 0.1523], [0.1402, 0.1402, 0.1402])
    ]),
    'test': transforms.Compose([
        NewPad(),
        transforms.Resize(wbconfig.input_size),
        transforms.CenterCrop(wbconfig.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.1523, 0.1523, 0.1523], [0.1402, 0.1402, 0.1402])
    ]),
}


'''Step 1: Build up dataset and dataloader'''

data_dir = "/mnt/HDD3/Users/dasleo/data/MURA-v1.1/all/"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                          data_transforms[x]) for x in ['train', 'val','test']}

dataloaders_dict = {}
dataloaders_dict['train'] = torch.utils.data.DataLoader(image_datasets['train'],
                                               batch_size=wbconfig.batch_size,
                                               shuffle=True,
                                               num_workers=4)
dataloaders_dict['val'] = torch.utils.data.DataLoader(image_datasets['val'],
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=2)
dataloaders_dict['test'] = torch.utils.data.DataLoader(image_datasets['test'],
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=2)

'''Step 2: Set up model'''
mura_model = EfficientNet.from_pretrained('efficientnet-b5',num_classes=2)
for _,param in mura_model.named_parameters():
    param.requires_grad = False


'''Step 6: Obtain Preliminary Classification Report'''
mura_model.load_state_dict(torch.load('/media/dasleo/LEOHDD/models/AnimalRadV2/MURA/7.h5'))
y_true,y_pred = test(model = mura_model,dataloaders = dataloaders_dict, device = device)
report = classification_report(y_true=y_true,y_pred=y_pred,target_names=['Abnormal','Normal'])
report_save_path = './MURA/log/Epoch_07_Test_Report.txt'
with open(report_save_path,"w") as file:
    file.write(report)