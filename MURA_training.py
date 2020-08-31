import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from efficientnet_pytorch import EfficientNet
import subprocess

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
num_epochs = 200
input_size = 480
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

model_names=["resnet50",
            "efficientnet-b5",
            "densenet201"]

def train_model(model, dataloaders, criterion, optimizer,
                num_epochs, text_path):
    print(f'Text file is saved to {text_path}')
    with open(text_path,"w") as file:
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
        with open(text_path,"a") as file:
            file.write(out_text)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
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
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            out_text ='{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)+"\n"
            with open(text_path,"a") as file:
                file.write(out_text)
            if phase == "val":
                val_acc_history.append(epoch_acc)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_acc_history.append(epoch_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    out_text = "\n"+'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + '\n' + 'Best val Acc: {:4f}'.format(best_acc) +'\n'
    with open(text_path,"a") as file:
        file.write(out_text)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history,train_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def initialize_model(structure,
                     num_classes,
                     feature_extract,
                     use_pretrained=True):
    model_ft = None
    print(f"Initialising model and set out_features to {num_classes}")
    if structure == "efficientnet-b5":
        model_ft = EfficientNet.from_pretrained('efficientnet-b5',num_classes=num_classes)
        set_parameter_requires_grad(model_ft,
                                feature_extracting = feature_extract)
        model_ft._fc = nn.Linear(in_features=2048,
                                out_features=2,
                                bias=True)
    if structure == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,
                                feature_extracting = feature_extract)
        model_ft.fc = nn.Linear(in_features=2048,
                                out_features=2,
                                bias=True)
    if structure == "densenet201":
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,
                                feature_extracting = feature_extract)
        model_ft.classifier = nn.Linear(in_features=1920,
                                        out_features=2,
                                        bias=True)
    return model_ft

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "/mnt/HDD3/Users/dasleo/data/MURA_Hum/"
path = "/mnt/HDD3/Users/dasleo/models/MURA_TVT/"
feature_extract=False
for model_name in model_names:
    print("*"*25,"\n",f"Current model is \t{model_name}\n")
    model_ft = initialize_model(structure=model_name,num_classes=2,feature_extract=feature_extract)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    if model_name=="resnet50":
        batch_size = 20
    if model_name=="efficientnet-b5":
        batch_size = 2
    if model_name=="densenet201":
        batch_size = 7
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=4) for x in ['train', 'val']}
    print(f"Model saving directory is \t{path}\n")
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    i=0
    if feature_extract:
        params_to_update = []
        for param_name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                i+=1
                print("\t",param_name,"\t",i)
    else:
        print("All Parameters are updated!\n")
    optimizer_ft = optim.Adam(params_to_update,lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    model_ft.to(device)
    out_text_path = path+model_name+".txt"
    #print(out_text_path)
    model_ft,val_hist,train_hist = train_model(model_ft,
                                               dataloaders_dict,
                                               criterion,
                                               optimizer_ft,
                                               num_epochs= num_epochs,
                                               text_path = out_text_path)
    model_save_path = path+model_name+"_best.h5"
    #print(model_save_path)
    torch.save(model_ft.state_dict(),model_save_path)
    fig_save_path = path+model_name+"_accuracy.png"
    vals = []
    vals = [h.cpu().numpy() for h in val_hist]
    trains=[]
    trains = [h.cpu().numpy() for h in train_hist]
    fig = plt.title("Validation and Training Accuracy of "+model_name)
    fig = plt.xlabel("Epochs")
    fig = plt.ylabel("Accuracy")
    fig = plt.plot(range(1,num_epochs+1),vals,label="Validation")
    fig = plt.plot(range(1,num_epochs+1),trains,label="Training")
    fig = plt.ylim((0,1.))
    fig = plt.xticks(np.arange(1, num_epochs+1, 20))
    fig = plt.legend()
    fig.figure.savefig(fig_save_path)
    fig.figure.clear()
