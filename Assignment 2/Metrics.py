# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:57:42 2024

@author: Veron
"""

# Shows the Accuracy, f1-score and confusion matrix
#Import packages
import torch # pytorch main library
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import torchvision # computer vision utilities
from torchvision import *

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import os
import re
import torchmetrics
from torchmetrics.classification import MultilabelConfusionMatrix

##-----------------------------------------------------------------------------------------------------------##
#-------------- Metrics -------------#

# F1- SCORE
def metrics_eval(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids']#.to(device)
            attention_mask = batch['attention_mask']#.to(device)
            labels = batch['label']#.to(device)
            images = batch['image']#.to(device)

            outputs = model(images, input_ids, attention_mask)
           
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        # convert to tensors
        y_pred_tensor = torch.tensor(y_pred)
        y_true_tensor = torch.tensor(y_true)
        # Calculating Recall, Precision, F1 Score, Accuracy
        TP= ((y_pred_tensor == 1)&(y_true_tensor == 1)).sum().item()
        FP= ((y_pred_tensor == 1)&(y_true_tensor == 0)).sum().item()
        TN= ((y_pred_tensor == 0)&(y_true_tensor == 0)).sum().item()
        FN= ((y_pred_tensor == 0)&(y_true_tensor == 1)).sum().item()
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 =2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy =100 * correct / total
        print(f'F1-SCORE: {f1}')
        print(f'Accuracy of the network on the 10000 test images: {accuracy} %')
        
        #CONFUSION MATRIX
        metric = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=4)
        metric(y_pred_tensor, y_true_tensor)
        fig_, ax_ = metric.plot()

##-----------------------------------------------------------------------------------------------------------##