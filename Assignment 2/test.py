# -*- coding: utf-8 -*-
import os
# os.system('pip install transformers')
# os.system('pip install torchmetrics')
# os.system('pip install torchviz')


# Import necessary functions from python files
# Model, Custom dataset and data extraction function
from Data_and_Model import read_text_files_with_labels, CustomDataset, GarbageModel

# Metrics
from Metrics import metrics_eval


#---------- Importing useful packages --------------#
import torch # pytorch main library
import glob
import torchvision # computer vision utilities
import torchvision.transforms as transforms # transforms used in the pre-processing of the data
from torchvision import *

from PIL import Image
from torchvision.models import resnet18, resnet50
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

#import hiddenlayer as hl
#from torchviz import make_dot

import time
import copy
import re
#from torchmetrics.classification import MultilabelConfusionMatrix

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# set paths to retrieve data
TEST_PATH = r"Assignment 2/Submit/CVPR_2024_dataset_Test" # Prediction images path

##-----------------------------------------------------------------------------------------------------------##
# Set the hyperparameters
batch_size = 256 # Change Batch Size o
learning_rate = 1e-4
num_workers = 2
nepochs = 35 #"Use it to change iterations"
best_loss = 1e+20
print(f'batch_size = {batch_size}, learning_rate = {learning_rate} num_workers = {num_workers} , nepochs = {nepochs} , best_loss = {best_loss}')
##-----------------------------------------------------------------------------------------------------------##

##-----------------------------------------------------------------------------------------------------------##
#------------ Data Loadinga and Pre-processing -------------------#
# Convert the data to a PyTorch tensor
#Resnet 18 stats for transformation
torchvision_transform = transforms.Compose([transforms.Resize((224,224)),\
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225] )])
    
torchvision_transform_test = transforms.Compose([transforms.Resize((224,224)),\
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225])])

# ResNet 50 Stats
    
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # BERT Tokenizer for caption text
max_len = 24

#Dataset has been pre-split, load all data sets.
test_dataset = CustomDataset(TEST_PATH, max_len, tokenizer, transform=torchvision_transform_test)


# Get the data loader for the test set
predloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

PATH = './garbage_net.pth'

# Using the metrics function to evaluate the model.
net = GarbageModel(4, (3,224,224), False)
net.load_state_dict(torch.load(PATH))


#PREDICTIONS
# Plot the misclassified samples
# label to word
def bin(label):
    if label == 0:
        color = 'BLACK'
    elif label == 1:
        color = 'BLUE'
    elif label == 2:
        color = 'GREEN'
    elif label == 3:
        color = 'OTHER'
    return color

def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(224, 224), cmap='gray')
    label = bin(data_sample[1])
    predicted = bin(data_sample[2])
    plt.title(str(data_sample[3])+'\n Predicted Bin= '+ predicted + ' Actual Bin= ' + label)

count = 0
predicted = 0
labels = 0
net.eval()
for batch in predloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    label = batch['label']
    images = batch['image']
    text = batch['text']
    z = net(images, input_ids, attention_mask)
    _, ypred = torch.max(z.data, 1)
    predicted = ypred.cpu().numpy()
    labels= label.cpu().numpy()
    if predicted != labels:
        plt.figure()
        show_data((images, label, ypred, text))
        plt.show()
        count += 1
    if count >= 5:
        break