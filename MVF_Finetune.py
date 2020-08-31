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
import pandas as pd

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
num_epochs = 100
input_size = 480
feature_extract = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

strategy_names=["Direct_Finetune",
                "MURA_Anim_Finetune",
                "Direct_FeatExt",
                "MURA_Anim_FeatExt"]
model_names=["resnet50",
            "efficientnet-b5",
            "densenet201"]
def run_bash(command):
    process = subprocess.Popen(command.split(),stdout=subprocess.PIPE)
    output,error = process.communicate()
    return output,error

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
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)
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
            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
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
col_names = []
for model_name in model_names:
    for strategy in strategy_names:
        col_names.append(model_name+"_"+strategy)
accuracy_data = pd.DataFrame(columns=col_names)
data_dir = "/mnt/HDD3/Users/dasleo/data/unsplit_long_bone/"
for model_name in model_names:
    for strategy in strategy_names:
        model_out_put_path = "/mnt/HDD3/Users/dasleo/models/round2/Cross_Validation/"+model_name+"/"+strategy+"/"
        command = "mkdir -p "+model_out_put_path
        output,error = run_bash(command)
test_results={}
for cv_num in np.arange(0,10):
    print("\n","-"*20)
    print(f"Current cross validation number is\t{cv_num}")
    current_cv_results = "CV_"+str(cv_num)
    test_results[current_cv_results] = []
    cv_dir = data_dir+"CV_"+str(cv_num)+"/"
    image_datasets = {x: datasets.ImageFolder(os.path.join(cv_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    for model_name in model_names:
        print(f"\n\nCurrent model is\t{model_name}")
        if model_name=="resnet50":
            batch_size = 20
        if model_name=="efficientnet-b5":
            batch_size = 2
        if model_name=="densenet201":
            batch_size = 7
        for strategy in strategy_names:
            model_out_put_path = "/mnt/HDD3/Users/dasleo/models/round2/Cross_Validation/"+model_name+"/"+strategy+"/"
            model_ft=None
            print(f'Current Strategy is\t{strategy}!')
            feature_extract = False
            if "FeatExt" in strategy:
                feature_extract = True
                current_batch_size=batch_size*2
            if ("FeatExt" in strategy) == False:
                current_batch_size = batch_size
            print(f"Feature extracting status is {feature_extract}\nBatch size is {current_batch_size}")
            model_ft = initialize_model(structure=model_name,num_classes=2,feature_extract=feature_extract)
            dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=current_batch_size,
                                                       shuffle=True,
                                                       num_workers=4) for x in ['train', 'val']}
            if "MURA_Anim" in strategy:
                path = "/mnt/HDD3/Users/dasleo/models/round2/MURA"+"/"+model_name+"/"+model_name+"_best.h5"
                model_ft.load_state_dict(torch.load(path))
                print(f"State Dictionary loaded from {path}")
                if "FeatExt" in strategy:
                    print("Reinitialising fully connected layers!")
                    if model_name == "resnet50":
                        model_ft.fc = nn.Linear(in_features=2048,
                                                out_features=2,
                                                bias=True)
                    if model_name == "efficientnet-b5":
                        model_ft._fc = nn.Linear(in_features=2048,
                                                out_features=2,
                                                bias=True)
                    if model_name == "densenet201":
                        model_ft.classifier = nn.Linear(in_features=1920,
                                                        out_features=2,
                                                        bias=True)
            params_to_update = model_ft.parameters()
            print("Params to learn:")
            i=0
            if feature_extract == True:
                params_to_update = []
                for param_name,param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        i+=1
                        print("\t",param_name,"\t",i)
            if feature_extract == False:
                print("All Parameters are updated!")
            optimizer_ft = optim.Adam(params_to_update,lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            criterion = nn.CrossEntropyLoss()
            model_ft.to(device)
            print("Model Sent to GPU!\nStart Training!\n\n")
            out_text_path = model_out_put_path+"CV_"+str(cv_num)+".txt"
            model_ft,val_hist,train_hist = train_model(model_ft,
                                               dataloaders_dict,
                                               criterion,
                                               optimizer_ft,
                                               num_epochs= num_epochs,
                                               text_path = out_text_path)
            test_results[current_cv_results].append(max(val_hist))
            model_save_path = model_out_put_path+"CV_"+str(cv_num)+"_best.h5"
            print(f"Best model for current CV is saved to {model_save_path}")
            torch.save(model_ft.state_dict(),model_save_path)
            print("Best Model Saved!!")
            fig_save_path = model_out_put_path+"CV_"+str(cv_num)+"_acces.png"
            vals = []
            vals = [h.cpu().numpy() for h in val_hist]
            trains=[]
            trains = [h.cpu().numpy() for h in train_hist]
            fig = plt.title("Validation and Training Accuracy of Densenet 201")
            fig = plt.xlabel("Epochs")
            fig = plt.ylabel("Accuracy")
            fig = plt.plot(range(1,num_epochs+1),vals,label="Validation")
            fig = plt.plot(range(1,num_epochs+1),trains,label="Training")
            fig = plt.ylim((0,1.))
            fig = plt.xticks(np.arange(1, num_epochs+1, 20))
            fig = plt.legend()
            fig.figure.savefig(fig_save_path)
            fig.figure.clear()
    accuracy_data.loc[len(accuracy_data),:]=test_results[current_cv_results]
accuracy_data.to_csv("./out/Round_2_Val_Accs.csv",encoding='utf-8',index=False)
