# Important Modules
from matplotlib import pyplot as plt
from random import randint
import numpy as np
import json
from PIL import Image
from datetime import datetime
import csv
import torch 
from torchvision import transforms, models 
from torch.utils.data import DataLoader 

from torchvision.models import resnet50, ResNet50_Weights, convnext_small, ConvNeXt_Small_Weights, inception_v3, Inception_V3_Weights, efficientnet_b7, EfficientNet_B7_Weights

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import os

#------------------- Utility Class and Functions -------------------------#
# All are pytorch dunctions frm the speedplus utils
# https://gitlab.com/EuropeanSpaceAgency/speedplus-utils/-/blob/main/utils.py?ref_type=heads
# dataset
class PyTorchSatellitePoseEstimationDataset(Dataset):
    """ SPEED dataset that can be used with DataLoader for PyTorch training. """
    def __init__(self, split, speed_root, transform=None):
        if split in {'train', 'validation'}:
            self.image_root = os.path.join(speed_root, 'synthetic', 'images')
            with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                label_list = json.load(f)
        else:
            self.image_root = os.path.join(speed_root, split, 'images')
            with open(os.path.join(speed_root, split, 'test.json'), 'r') as f:
                label_list = json.load(f)
        self.sample_ids = [label['filename'] for label in label_list]
        self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
            for label in label_list}
        self.split = split
        self.transform = transform
    def __len__(self):
        return len(self.sample_ids)
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_name = os.path.join(self.image_root, sample_id)
        # note: despite grayscale images, we are converting to 3 channels here,
        # since most pre-trained networks expect 3 channel input
        pil_image = Image.open(img_name).convert('RGB')
        q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
        y = np.concatenate([q, r])
        if self.transform is not None:
            torch_image = self.transform(pil_image)
        else:
            torch_image = pil_image
        return torch_image, y
#--------------------------------------------#

#------------------- Model -------------------------#
# Define the model
class PoseResNetModel(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.input_shape = shape
        self.feature_extractor = resnet50(weights='DEFAULT')
        n_features = self._get_conv_output(self.input_shape)
        self.bnorm = nn.BatchNorm1d(n_features)
        self.pose = nn.Linear(n_features, 7)


    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.feature_extractor(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, image, val):
       # Image feature layers
       image_features = self.feature_extractor(image) # RessNet50 transfer learning
       image_features = image_features.view(image_features.size(0), -1)
       image_features = self.bnorm(image_features)
       pose = self.pose(F.gelu(image_features))
       return pose


# ConvNext
class PoseConvNextmodel(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.input_shape = shape
        self.feature_extractor = convnext_small(weights='DEFAULT')
        # print('its running')
        # self.bnorm = nn.BatchNorm1d(1000) # 1000 gotten from # of classes of inception model https://pytorch.org/vision/main/_modules/torchvision/models/convnext.html#convnext_base
        # self.pose = nn.Linear(1000, 7)
        n_features = self._get_conv_output(self.input_shape)
        self.bnorm = nn.BatchNorm1d(n_features)
        self.pose = nn.Linear(n_features, 7)


    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.feature_extractor(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, image, val):
       # Image feature layers
       image_features = self.feature_extractor(image) # ConvNext transfer learning
       image_features = image_features.view(image_features.size(0), -1)
       image_features = self.bnorm(image_features)
       pose = self.pose(F.gelu(image_features))
       
       return pose

# InceptionNet
class PoseIceptionV3model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.input_shape = shape
        self.feature_extractor = inception_v3(weights='DEFAULT')
        self.bnorm = nn.BatchNorm1d(1000) # 1000 gotten from # of classes of inception model https://pytorch.org/vision/main/_modules/torchvision/models/inception.html#inception_v3
        self.pose = nn.Linear(1000, 7)


    def forward(self, image, val):
       # Image feature layers
       image_features = self.feature_extractor(image) # Inception V3 transfer learning
       if val == 'train':
           image_features = self.bnorm(image_features.logits)
       else:
            image_features = self.bnorm(image_features)
       pose = self.pose(F.gelu(image_features))
       
       return pose
    
#EfficientNet_B7_Weights
class PoseEfficientNettmodel(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.input_shape = shape
        self.feature_extractor = efficientnet_b7(weights='DEFAULT')
        self.bnorm = nn.BatchNorm1d(1000) # 1000 gotten from # of classes of inception model https://pytorch.org/vision/main/_modules/torchvision/models/efficientnet.html#efficientnet_b7
        self.pose = nn.Linear(1000, 7)

    def forward(self, image, val):
       # Image feature layers
       image_features = self.feature_extractor(image) # Efficientnet transfer learning
       image_features = self.bnorm(image_features)
       pose = self.pose(F.gelu(image_features))
       return pose

def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

class Camera:
    # """" Utility class for accessing camera parameters. """
    k = [
        [224, 0, 0],
        [0, 224, 0],
        [0, 0, 1],
    ]
    K = np.array(k)

def project(q, r):
    """ Projecting points to image frame to draw axes """
    # reference points in satellite frame for drawing axes
    p_axes = np.array([[0, 0, 0, 1],
                       [1, 0, 0, 1],
                       [0, 1, 0, 1],
                       [0, 0, 1, 1]])
    points_body = np.transpose(p_axes)
    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)
    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]
    # projection to image plane
    points_image_plane = Camera.K.dot(points_camera_frame)
    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y

def viz_pose(img, q, r):
    """ Visualizing image, with ground truth pose with axes projected to training image. """
    #ax = plt.gca()

    plt.imshow(img)
    xa, ya = project(q, r)
    plt.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=10, color='r')
    plt.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=10, color='g')
    plt.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=10, color='b')
    return