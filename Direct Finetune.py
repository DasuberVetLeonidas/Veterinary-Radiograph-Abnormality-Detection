from __future__ import print_function
from __future__ import division
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
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

data_dir = "/mnt/HDD2/Users/dasleo/data/Animal/image_data/"
model_name = "Resnet 50"
num_classes = 2
batch_size = 20
num_epochs = 100
feature_extract = False
input_size = 480
def train_model(model, dataloaders, criterion, optimizer,
                num_epochs,
                text_path = '/mnt/HDD2/Users/dasleo/models/resnet_50_anim_direct_FT/out_text.txt'):
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
        out_text = '\n' +'Epoch {}/{}'.format(epoch, num_epochs - 1) + '\n' + '-' * 10
        with open(text_path,"a") as file:
            file.write(out_text)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            out_text ="\n"+'{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
            with open(text_path,"a") as file:
                file.write(out_text)
            # deep copy the model
            if phase == "val":
                current_model_wts = copy.deepcopy(model.state_dict())
                model_name = str(epoch)+".h5"
                torch.save(current_model_wts,
                           os.path.join("/mnt/HDD2/Users/dasleo/models/resnet_50_anim_direct_FT/",
                                        model_name))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    out_text = "\n"+'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                             time_elapsed % 60) + '\n' + 'Best val Acc: {:4f}'.format(best_acc) +'\n'
    with open(text_path,"a") as file:
        file.write(out_text)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history,train_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes,feature_extract,use_pretrained=True):
    model_ft = None
    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft,
                                feature_extracting = feature_extract)
    num_ftrs=model_ft.fc.in_features
    model_ft.fc=nn.Sequential(
        nn.Linear(num_ftrs,int(num_ftrs/2)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(num_ftrs/2),num_classes))
    return model_ft

model_ft = initialize_model(num_classes,feature_extract,
                                     use_pretrained=True)
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
print("Initializing Datasets and Dataloaders...")
# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4) for x in ['train', 'val']}
# Detect if we have a GPU available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
#print("Params to learn:")
i=0
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            i+=1
            #print("\t",name,"\t",i)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            i+=1
            print("\t",name,"\t",i)
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update,lr=0.00001, betas=(0.9, 0.999),
                          eps=1e-08, weight_decay=0, amsgrad=False)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

model_ft, val_hist,train_hist = train_model(model_ft, dataloaders_dict,
                                            criterion, optimizer_ft,
                                            num_epochs=num_epochs)

torch.save(model_ft.state_dict(),
           os.path.join("/mnt/HDD2/Users/dasleo/models/resnet_50_anim_direct_FT/",
                        "best.h5"))
vals = []
vals = [h.cpu().numpy() for h in val_hist]
trains=[]
trains = [h.cpu().numpy() for h in train_hist]
plt.title("Validation and Training Accuracy of Resnet 50")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(range(1,num_epochs+1),vals,label="Validation")
plt.plot(range(1,num_epochs+1),trains,label="Training")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()
plt.savefig("/mnt/HDD2/Users/dasleo/models/resnet_50_anim_direct_FT/resnet_50_anim_direct_FT.png")
