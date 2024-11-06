# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:59:51 2024

@author: Veron
"""

# import os
# os.system('pip install torch torchvision torchaudio')
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
from torchvision.models import resnet18, resnet50, ResNet50_Weights
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

#import hiddenlayer as hl
from torchviz import make_dot

import time
import copy
import os
import re
from torchmetrics.classification import MultilabelConfusionMatrix

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
##-----------------------------------------------------------------------------------------------------------##
##-----------------------------------------------------------------------------------------------------------##
# set paths to retrieve data

TEST_PATH = r"Assignment 2/Submit/CVPR_2024_dataset_Test"
##-----------------------------------------------------------------------------------------------------------##
# Set the hyperparameters
batch_size = 16 # Change Batch Size o
learning_rate = 1e-3
num_workers = 0
nepochs = 5 #"Use it to chane iterations"
best_loss = 1e+20
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

    
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # BERT Tokenizer for caption text
max_len = 24

#Dataset has been pre-split, load all data sets.
test_dataset = CustomDataset(TEST_PATH, max_len, tokenizer, transform=torchvision_transform_test)


# Get the data loader for the train set
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# #Dataset has been pre-split, load all data sets.
# train_set_size = int(len(train_dataset))
# val_set_size = int(len(val_dataset))
# test_set_size = int(len(test_dataset))

# #classes = ('green', 'blue', 'black', 'other')
# print("Train set:", train_set_size)
# print("Val set:", val_set_size)
# print("Test set:", test_set_size)

# # visualize class distribution
# class_labels, class_counts = np.unique(train_dataset.labels, return_counts=True)

# plt.bar(class_labels, class_counts, color='blue')
# plt.xlabel('Class Label')
# plt.ylabel('Number of Examples')
# plt.title('Distribution of Examples Across Classes (Training Data)')
# plt.savefig("ClassDistribution.png") 
# plt.show()

# # Calculating Class weights
# num_classes = len(class_counts)

# class_weights = []
# for count in class_counts:
#     weight = 1 / (count / train_set_size)
#     class_weights.append(weight)


# class_weights = torch.FloatTensor(class_weights).to(device)

# # The file names will be extracted and used as labels
# # One- Hot Encoding is used to repressent labels for easy use of categorical cross- Entropy Loss.
# # Pre processing for ResNet-50. Inputs and output sizes,
# # ##-----------------------------------------------------------------------------------------------------------##

# # # get some random training images
# # train_iterator = iter(trainloader)
# # train_batch = next(train_iterator)

# # # Visualizing a sample image from dataset
# # plt.figure()
# # plt.imshow(train_batch['image'].numpy()[8].transpose(1,2,0)) # Convert tensor to numpy array
# # plt.show()



# ##-------------------------------------------------GARBAGE CLASSIFICATION----------------------------------------------------------##
# ### Testing the Model
# net = GarbageModel(4, (3,224,224), True)

# # # Visualize Model
# # yhat = net(train_batch['image'], train_batch['input_ids'], train_batch['attention_mask'])
# # # Backward propagation
# # make_dot(yhat, params = dict(list(net.named_parameters()))).render("GarbageClassification_torchviz", format="png")
# # #Forward
# # modelviz_transform= [hl.transforms.Prune('Constant')] #removes Constant nodes
# # graph = hl.build_graph(net, train_batch.text, transforms=modelviz_transform)
# # #graph.theme
# # graph.save('GarbageClass_hiddenlayer', format='png')

# net.to(device)
# #------- Training Parameters ---------#
# # Loss Function
# criterion = nn.CrossEntropyLoss(weight = class_weights) # Loss function
# optimizer = torch.optim.AdamW(net.parameters(), lr = 0.001) # Optimizer used for training
# # Decay LR by a factor of 0.1 every 7 epochs
# scheduler = ExponentialLR(optimizer, gamma=0.9)

# ##---------------------------Main------------##
# PATH = './garbage_net.pth' # Path to save the best model
# e = []
# trainL= []
# valL =[]
# for epoch in range(nepochs):  # loop over the dataset multiple times
#     # Training Loop
#     net.train()
#     train_loss = 0.0
#     for i, batch in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)
#         images = batch['image'].to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(images, input_ids, attention_mask)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#     print(f'{epoch + 1},  train loss: {train_loss / i:.3f},', end = ' ')
#     trainL.append(train_loss)
#     scheduler.step()

# #---------------Validation----------------------------#
#     net.eval()
#     val_loss = 0.0
#     # since we're not training, we don't need to calculate the gradients for our outputs
#     with torch.no_grad():
#         for i, batch in enumerate(valloader, 0):
#             # get the inputs; data is a list of [inputs, labels] 
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['label'].to(device)
#             images = batch['image'].to(device)

#             outputs = net(images, input_ids, attention_mask)
#             loss = criterion(outputs, labels)

#             val_loss += loss.item()
#         print(f'val loss: {val_loss / i:.3f}')

#         # Save best model
#         if val_loss < best_loss:
#             print("Saving model")
#             torch.save(net.state_dict(), PATH)
#             best_loss = val_loss
#     valL.append(val_loss)
#     e.append(epoch)

# # Visualize training and Loss functions
# plt.figure()
# plt.plot(e, trainL, label = "Train loss")
# plt.plot(e, valL, label = "Val loss")
# plt.xlabel("Epoch (iteration)")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid()
# plt.savefig("lossfunction.png") 
# plt.show()
# print('Finished Training')

##------------------------------TESTING-----------------------------------------------------------------------------##
# Using the metrics function to evaluate the model.
PATH = './garbage_net.pth'
net = GarbageModel(4, (3,224,224), False)
net.load_state_dict(torch.load(PATH))
# metrics_eval(net, testloader, device)
##-----------------------------------------------------------------------------------------------------------##


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
    plt.imshow(data_sample[0].transpose(1,2,0))
    label = bin(data_sample[1])
    predicted = bin(data_sample[2])
    plt.title(data_sample[3]+'\n Predicted Bin= '+ predicted + ' Actual Bin= ' + label)

# predicted = 0
# labels = 0
net.eval()
with torch.no_grad():
    counter = 0
    for batch in testloader:
        if counter > 5: #end after 5 pictures shown
            break
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']
        images = batch['image']
        text = batch['text']
        z = net(images, input_ids, attention_mask)
        _, ypred = torch.max(z.data, 1)
        predicted = ypred.cpu().numpy()
        labels= label.cpu().numpy()
        image = images.numpy()

        for i in range(len(predicted)):
            if predicted[i] != labels[i]:
                plt.figure()
                show_data((image[i], labels[i], predicted[i], text[i]))
                plt.show()
                counter += 1
            if counter >5:
                break