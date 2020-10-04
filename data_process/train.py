import datetime
import torch 
import numpy as np
import pandas as pd
import json
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import f1_score

plt.ion()   # interactive mode

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model,data_loaders,datasets,criterion,optimizer,num_epochs,logging=True,save_path=None ):
    since=time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_F1=0.0
    for epoch in range(num_epochs):
        print(f"epoch: {epoch}/{num_epochs}")
        print("-"*10)
        for phase in ["train","validation"]:
            if phase=="train":
                model.train()
            else:
                model.eval()
            running_loss=0
            running_correct=0    
            for inputs,labels in data_loaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                # zeroing the parameter gradinets
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds=outputs>=0.5
                    loss = criterion(outputs, labels)
                    if phase=="train":
                        loss.backward()
                        criterion.step()
                running_loss+=loss.item()*inputs.size(0)
                running_correct+=torch.sum(preds==labels)     
            epoch_loss=running_loss/len(datasets[phase])  
            epoch_acc=running_correct/len(datasets[phase])  
            print(f"{phase} Loss: {epoch_loss:.3f} acc: {epoch_acc:.3f}")  
        if (phase=="validation")and(epoch_acc>best_acc):
            best_acc=epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model          




