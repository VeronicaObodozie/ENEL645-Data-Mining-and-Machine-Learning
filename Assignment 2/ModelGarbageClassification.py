# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:41:09 2024

@author: Veron
"""

"---------------------------------------"
"Garbage Classification Model"

# Importing Packages
import torch # pytorch main library 
import torchvision # computer vision utilities
import torchvision.transforms as transforms # transforms used in the pre-processing of the data
from torchvision import *
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim

import time
import copy
import os

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


# Function to get thge statistics of a dataset
def get_dataset_stats(data_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in data_loader:
        data = data[0] # Get the images to compute the stgatistics
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        
    mean /= nb_samples
    std /= nb_samples
    return mean,std

# functions to show an image
def imshow(img,stats):
    img = img *stats[1] + stats[0]     # unnormalize
    npimg = img.numpy() # convert the tensor back to numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
    

batch_size = 128 = " Change Batch Size o"
learning_rate = 1e-3

transform = transforms.Compose(
    [transforms.ToTensor()]) # Convert the data to a PyTorch tensor

#Dataset has been pre-split, load all data sets.
-devset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform = transform)
[from second ref] train_dataset = datasets.ImageFolder(root='/input/fruits-360-dataset/fruits-360/Training', transform=transforms)
test_dataset = datasets.ImageFolder(root='/input/fruits-360-dataset/fruits-360/Test', transform=transforms)

"Create These"
-trainSet=
-valnSet =
-testSet
train_set_size = int(len(trainSet))
val_set_size = int(len(validationSet))
test_set_size = int(len(testSet))

# Get the data loader for the train set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

="Dataloader also needs creation"
# Comopute the statistics of the train set
stats = get_dataset_stats(trainloader)
print("Train stats:", stats)
# Pre-processing transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((stats[0]), (stats[1]))])

classes = ('green', 'blue', 'black', 'other')

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images[:8]), stats)
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))


#------------Data Import and Pre-processing---------#
# Accessing data from cluster- data was pre-downloaded

# Normalizing Data using Standardization.

# Displaying Normalized data
#Displaying the normalized train set
plt.scatter(X_train[:,0],X_train[:,1],c=Y_train)
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()
plt.title("Normalized Train Set")
plt.show()


# One- Hot Encoding is used to repressent labels for easy use of categorical cross- Entropy Loss.


##----------Transfer Learning----------------##
# ResNet-50 being used as a fixed feature extractor and the last fully connected layer will be trained

#----------- Defining Model. -------------#
##  Transfer Learning of ResNet-50
net = models.resnet18(pretrained=True)# Calling a pre-trained ResNet-50 model
net = net.cuda() if device else net
net 

net.to(device)

# Loss Function
criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # Optimizer used for training

# Number of filters
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 128)
net.fc = net.fc.cuda() if use_cuda else net.fc

#---------- Training--------------#

# Batch size is the number of elements in the training dataset folder
# Number of epochs

nepochs = 20 = "Use it to chane iterations"
PATH = './cifar_net.pth' # Path to save the best model

best_loss = 1e+20
for epoch in range(nepochs):  # loop over the dataset multiple times
    # Training Loop
    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f'{epoch + 1},  train loss: {train_loss / i:.3f},', end = ' ')
    
    val_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
        print(f'val loss: {val_loss / i:.3f}')
        
        # Save best model
        if val_loss < best_loss:
            print("Saving model")
            torch.save(net.state_dict(), PATH)
        
print('Finished Training')


#---------------- Validation ---------#

# Load the best model to be used in the test set
net = Net()
net.load_state_dict(torch.load(PATH))





# Printing training Set accuracy and Cross Entropy Loss
# Images saved for future considerations

#-------------- Test --------------#
dataiter = iter(testloader)
images, labels = next(dataiter)


# print images
imshow(torchvision.utils.make_grid(images[:4]), stats)
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

#------------- Metrics ------------#

==""//====
correct = 0
total = 0

#def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()
"Check how to define Accuracy"
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


##
#plot Accuracy
fig = plt.figure(figsize=(20,10))
plt.title("Train-Validation Accuracy")
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')

def visualize_model(net, num_images=4):
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    for i, data in enumerate(test_dataloader):
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        _, preds = torch.max(outputs.data, 1)
        preds = preds.cpu().numpy() if use_cuda else preds.numpy()
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(2, num_images//2, images_so_far)
            ax.axis('off')
            ax.set_title('predictes: {}'.format(test_dataset.classes[preds[j]]))
            imshow(inputs[j])
            
            if images_so_far == num_images:
                return 

plt.ion()
visualize_model(net)
plt.ioff()

# References
CNN Minst tutorial
https://www.pluralsight.com/resources/blog/guides/introduction-to-resnet
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

