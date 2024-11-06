# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:58:00 2024

@author: Veron
"""

# useful packages
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
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import os
import re

# Detailing the Customized dataset, Garbage classiication model and the function to extract text, image path and labels
##-----------------------------------------------------------------------------------------------------------##
#----------- Customized Dataset -------------#
def read_text_files_with_labels(path):
    texts = []
    labels = []
    image_paths = []
    class_folders = sorted(os.listdir(path))  # Assuming class folders are sorted
    label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}
    # Loop through folders and files to get images, labels and caption.
    for class_name in class_folders:
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            file_names = os.listdir(class_path)
            for file_name in file_names:
                file_path = os.path.join(class_path, file_name)
                if os.path.isfile(file_path):
                    file_name_no_ext, _ = os.path.splitext(file_name)
                    text = file_name_no_ext.replace('_', ' ')
                    text_without_digits = re.sub(r'\d+', '', text)
                    texts.append(text_without_digits)
                    labels.append(label_map[class_name])
                    image_paths.append(file_path)

    return np.array(texts), np.array(image_paths), np.array(labels)

class CustomDataset(Dataset):
    def __init__(self, root_dir, max_len, tokenizer, transform= None):
        self.texts, self.image_paths, self.labels = read_text_files_with_labels(root_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Handling image Data
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
          image = self.transform(image)

        return {'image': image, 'label': torch.tensor(label, dtype=torch.long), 'text':text, 'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}

##-----------------------------------------------------------------------------------------------------------##
#----------- Defining Model. -------------#
# Model
class GarbageModel(nn.Module):
    def __init__(self,  num_classes, input_shape, transfer=False):
        super().__init__()

        self.transfer = transfer
        self.num_classes = num_classes
        self.input_shape = input_shape

        # Image feature extraction layer
        self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if self.transfer:
            # layers are frozen by using eval()
            self.feature_extractor.eval()
            # freeze params
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        #self.drop1 = nn.Dropout(0.4)
        n_features = self._get_conv_output(self.input_shape)
        self.image_features = nn.Linear(n_features, 512)
        self.bnorm1 = nn.BatchNorm1d(512)
        #handling Text
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc1 = nn.Linear(self.distilbert.config.hidden_size, 512)
        self.bnorm2 = nn.BatchNorm1d(512)
        #self.drop = nn.Dropout(0.2)
        ## Setting up Model Classifier
        #self.dense = nn.Linear(256, 64)
        #self.bnorm = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(1024, num_classes)

# this gets the number of filters from the feature extractor output
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.feature_extractor(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # will be used during inference
    def forward(self, image, input_ids, attention_mask):
       # Image feature layers
       image_features = self.feature_extractor(image) # RessNet50 transfer learning
       #image_features = self.drop1(image_features) #dropout
       image_features = image_features.view(image_features.size(0), -1)
       image_features = self.image_features(image_features)
       image_features = F.relu(self.bnorm1(image_features))

       # Text feature layers
       text_features = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, -1, :]
       text_features = self.fc1(text_features)
       text_features = F.relu(self.bnorm2(text_features))
       #text_features = self.drop(text_features)

       # merged features for classifier
       features = torch.cat((image_features, text_features), dim=-1)
       #features = self.dense(features)
       #features = F.relu(features)
       #features =  self.drop1(features)
       x = self.classifier(features)
       x= F.log_softmax(x, dim=1)
       return x

