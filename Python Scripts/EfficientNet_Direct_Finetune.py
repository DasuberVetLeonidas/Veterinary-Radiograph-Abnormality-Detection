
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
model_name = "Efficientnetb5"


# ### Hyperparameters

# In[ ]:


cv_dir = "/media/dasleo/LEOHDD/data/AnimalDatasetV2CVs/"

hyperparameter_defaults = dict(
    input_size = 1024,
    batch_size = 5,
    training_depth = 19,
    num_epochs = 100,
    learning_rate = 0.0006007,
    betas1 = 0.7807,
    betas2 = 0.905,
    eps = 0.000001976,
    amsgrad = False,
)

wandb.init(project="Direct Finetune with Dataset",config=hyperparameter_defaults)
wbconfig = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''Compute balanced classweight using scikit-learn'''
from sklearn.utils import class_weight
meta = pd.read_csv('./meta.csv')
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(meta['Normal_Abnormal']),
                                                 meta['Normal_Abnormal'])
class_weights = torch.tensor([class_weights],dtype=torch.float)

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

def train_model(model, dataloaders, criterion, optimizer,
                num_epochs, log_path, model_save_path,scheduler,wbconfig=wbconfig):
    print(f'Training log is saved to {log_path}')
    with open(log_path,"w") as file:
        file.write("\n")
    since = time.time()
    val_acc_history = []
    train_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        out_text = ""
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        out_text = '\n' +'Epoch {}/{}'.format(epoch, num_epochs - 1) + '\n' + '-' * 10 +'\n'
        with open(log_path,"a") as file:
            file.write(out_text)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                training_size = len(dataloaders['train'].dataset)
                total_steps = training_size//wbconfig.batch_size
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_steps -= 1
                    wandb.log({"Training Loss":loss})
                    wandb.log({"Steps left in current epoch":total_steps})
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'train':
                wandb.log({"Epoch Training Accuracy":epoch_acc*100})
            if phase == 'val':
                wandb.log({"Epoch Validation Accuracy":epoch_acc*100})
                wandb.log({"Epoch Validation Loss":epoch_loss})
            out_text ='{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)+"\n"
            with open(log_path,"a") as file:
                file.write(out_text)
            save_path = model_save_path + '/' + str(epoch) + '.h5'
            torch.save(model.state_dict(),save_path)
            if phase == "val":
                val_acc_history.append(epoch_acc)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_acc_history.append(epoch_acc)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        wandb.log({"Current LR":lr})
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    out_text = "\n"+'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + '\n' + 'Best val Acc: {:4f}'.format(best_acc) +'\n'
    with open(log_path,"a") as file:
        file.write(out_text)
    model.load_state_dict(best_model_wts)
    save_path = model_save_path + '/' + 'best.h5'
    return model, val_acc_history,train_acc_history


def test(model, dataloaders, device):
    labels = []
    preds = []
    total = 0
    num_correct = 0
    for _,param in model.named_parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    for inputs,label in dataloaders['test']:
        inputs = inputs.to(device)
        total += 1
        outputs = model(inputs)
        _,pred = torch.max(outputs,1)
        pred = pred.to('cpu').detach()  
        labels.append(int(label.detach()))
        preds.append(int(pred))
    print(f"Label is\t{labels} prediction is\t{preds}")
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



# In[ ]:


## Different CV Numbers
models = {}

for cv in np.arange(1,5+1):

    if cv == 0:
        print(f"CV{cv} Training is skipped.")
        trained_model = EfficientNet.from_pretrained('efficientnet-b5',num_classes=2)
        state_dict_path = '/media/dasleo/LEOHDD/models/AnimalRadV2/Direct_Finetune/CV' + str(cv) +'/best.h5'
        data_dir = cv_dir+'CV'+str(cv)
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                                  data_transforms[x]) for x in ['train', 'test']}
        dataloaders_dict = {}
        dataloaders_dict['test'] = torch.utils.data.DataLoader(image_datasets['test'],
                                                           batch_size=1,
                                                           shuffle=False,
                                                           num_workers=2)
        trained_model.load_state_dict(torch.load(state_dict_path,map_location='cuda:0'))
        y_true,y_pred = test(model = trained_model,dataloaders = dataloaders_dict, device = device)
        report = classification_report(y_true=y_true,y_pred=y_pred)
        report_save_path = './Direct_Finetune/log/CV'+str(cv)+'_Report.txt'
        with open(report_save_path,"w") as file:
            file.write(report)
    else:
        '''Step 1: Build up dataset and dataloader'''
        data_dir = cv_dir+'CV'+str(cv)
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                                  data_transforms[x]) for x in ['train', 'test']}
        [train_set,val_set] = random_split(image_datasets['train'],
                                           [int(len(image_datasets['train'])*0.9),
                                            int(len(image_datasets['train'])*0.1)])
        #print("The class labels and indexes are",'\t',image_datasets['train'].class_to_idx)
        dataloaders_dict = {}
        dataloaders_dict['train'] = torch.utils.data.DataLoader(train_set,
                                                       batch_size=wbconfig.batch_size,
                                                       shuffle=True,
                                                       num_workers=4)
        dataloaders_dict['val'] = torch.utils.data.DataLoader(val_set,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           num_workers=2)
        dataloaders_dict['test'] = torch.utils.data.DataLoader(image_datasets['test'],
                                                           batch_size=1,
                                                           shuffle=False,
                                                           num_workers=2)
        
        '''Step 2: Set up model'''
        models[cv] = EfficientNet.from_pretrained('efficientnet-b5',num_classes=2)
        for _,param in models[cv].named_parameters():
            param.requires_grad = False
        for name,param in models[cv].named_parameters():
            if name.split('.')[1].isdigit():
                if int(name.split('.')[1]) > wbconfig.training_depth:
                    param.requires_grad = True
        models[cv]._conv_head.weight.requires_grad = True
        models[cv]._bn1.weight.requires_grad = True
        models[cv]._bn1.bias.requires_grad = True
        models[cv]._fc.weight.requires_grad = True
        models[cv]._fc.bias.requires_grad = True
        # collect parameters to be trained
        params_to_update = []
        for name,param in models[cv].named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        
        '''Step 3: Build required stuff for training helper function'''
        wandb.watch(models[cv],log='all')
        models[cv].to(device)
        dataloaders = dataloaders_dict
        criterion = nn.CrossEntropyLoss(weight = class_weights).to(device)
        optimizer = optim.Adam(params_to_update,
                              lr = wbconfig.learning_rate,
                              betas=(wbconfig.betas1,wbconfig.betas2),
                              eps=wbconfig.eps,
                              amsgrad=wbconfig.amsgrad)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
        num_epochs = wbconfig.num_epochs
        log_path = './Direct_Finetune/log/CV'+str(cv)+'_log.txt'
        model_save_path = './Direct_Finetune/CV'+str(cv)
        
        '''Step 4: Train model'''
        trained_model,val_acc_history,train_acc_history  = train_model(models[cv],
                                                                       dataloaders_dict, 
                                                                       criterion,
                                                                       optimizer,
                                                                       num_epochs,
                                                                       log_path,
                                                                       scheduler = scheduler,
                                                                       model_save_path=model_save_path)
        save_path = './Direct_Finetune/CV'+str(cv)+'/best.h5'
        torch.save(trained_model.state_dict(),save_path)
        
        '''Step 5: Draw Training History Figures'''
        fig = plt.title("Validation and Training Accuracy")
        fig = plt.xlabel("Epochs")
        fig = plt.ylabel("Accuracy")
        fig = plt.plot(range(1,num_epochs+1),val_acc_history,label="Validation")
        fig = plt.plot(range(1,num_epochs+1),train_acc_history,label="Training")
        fig = plt.ylim((0,1.))
        fig = plt.xticks(np.arange(1, num_epochs+1, 10))
        fig = plt.legend()
        fig_save_path = './Direct_Finetune/log/CV'+str(cv)+'_Train_Val_Log.png'
        fig.figure.savefig(fig_save_path)
        
        '''Step 6: Obtain Preliminary Classification Report'''
        y_true,y_pred = test(model = trained_model,dataloaders = dataloaders_dict, device = device)
        report = classification_report(y_true=y_true,y_pred=y_pred)
        report_save_path = './Direct_Finetune/log/CV'+str(cv)+'_Report.txt'
        with open(report_save_path,"w") as file:
            file.write(report)
        
        '''Final Step: purging before next CV'''
        del trained_model, image_datasets, dataloaders_dict, dataloaders, val_acc_history, train_acc_history, fig, report
