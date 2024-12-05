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

# class Camera:

#     """" Utility class for accessing camera parameters. """
    
#     speed_root = "Pose\speedplusv2/"
    
#     with open(os.path.join('Pose/speedplusv2/camera.json'), 'r') as f:
#         camera_params = json.load(f)

#     fx = camera_params['fx'] # focal length[m]
#     fy = camera_params['fy'] # focal length[m]
#     nu = camera_params['Nu'] # number of horizontal[pixels]
#     nv = camera_params['Nv'] # number of vertical[pixels]
#     ppx = camera_params['ppx'] # horizontal pixel pitch[m / pixel]
#     ppy = camera_params['ppy'] # vertical pixel pitch[m / pixel]
#     fpx = fx / ppx  # horizontal focal length[pixels]
#     fpy = fy / ppy  # vertical focal length[pixels]
#     k = camera_params['cameraMatrix']
#     K = np.array(k) # cameraMatrix
#     dcoef = camera_params['distCoeffs']


def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'synthetic', 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'synthetic', 'validation.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'sunlamp', 'test.json'), 'r') as f:
        sunlamp_image_list = json.load(f)

    with open(os.path.join(root_dir, 'lightbox', 'test.json'), 'r') as f:
        lightbox_image_list = json.load(f)

    partitions = {'validation': [], 'train': [], 'sunlamp': [], 'lightbox': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango_true'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['validation'].append(image['filename'])

    for image in sunlamp_image_list:
        partitions['sunlamp'].append(image['filename'])

    for image in lightbox_image_list:
        partitions['lightbox'].append(image['filename'])

    return partitions, labels


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
    x0, y0 = (points_camera_frame[0], points_camera_frame[1])
    
    # apply distortion
    dist = Camera.dcoef
    
    r2 = x0*x0 + y0*y0
    cdist = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2
    x1  = x0*cdist + dist[2]*2*x0*y0 + dist[3]*(r2 + 2*x0*x0)
    y1  = y0*cdist + dist[2]*(r2 + 2*y0*y0) + dist[3]*2*x0*y0
    
    # projection to image plane
    x = Camera.K[0,0]*x1 + Camera.K[0,2]
    y = Camera.K[1,1]*y1 + Camera.K[1,2]
    
    return x, y

class SatellitePoseEstimationDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='"Pose\speedplusv2"'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        if split=='train':
            img_name = os.path.join(self.root_dir, 'synthetic','images', img_name)
        elif split=='validation':
            img_name = os.path.join(self.root_dir, 'synthetic','images', img_name)
        elif split=='sunlamp':
            img_name = os.path.join(self.root_dir, 'sunlamp','images', img_name)
        elif split=='lightbox':
            img_name = os.path.join(self.root_dir, 'lightbox','images', img_name)
        else:
            print()
            # raise error?
        
        image = Image.open(img_name).convert('RGB')
        return image

    def get_pose(self, i=0):

        """ Getting pose label for image. """

        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def visualize(self, i, partition='train', ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        # no pose label for test
        if partition == 'train':
            q, r = self.get_pose(i)
            xa, ya = project(q, r)
            ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
            ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
            ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')

        return

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
        self.train = split == 'train'
        self.val = split == 'validation'
        if self.train:
            self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
                           for label in label_list}
        elif self.val:
            self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']}
                           for label in label_list}
        else:
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
        if self.train:
            q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
            y = np.concatenate([q, r])
        elif self.val:
            q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
            y = np.concatenate([q, r])
        else:
            q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
            y = np.concatenate([q, r]) #sample_id
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

    def forward(self, image):
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

    def forward(self, image):
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

    def forward(self, image):
       # Image feature layers
       image_features = self.feature_extractor(image) # Efficientnet transfer learning
       image_features = self.bnorm(image_features)
       pose = self.pose(F.gelu(image_features))
       return pose



#------------------------ Extracting Results---------------------------#
# Extraction code retrieved from: https://gitlab.com/EuropeanSpaceAgency/speedplus-utils/-/blob/main/submission.py?ref_type=heads
class SubmissionWriter:

    """ Class for collecting results and exporting submission. """

    def __init__(self):
        self.test_results = []
        self.real_test_results = []
        return

    def _append(self, filename, q, r, testdata, real):
        if real:
            self.real_test_results.append({'testdata' : testdata, 'filename': filename, 'q': list(q), 'r': list(r)})
        else:
            self.test_results.append({'testdata' : "synthetic", 'filename': filename, 'q': list(q), 'r': list(r)})
        return

    def append_test(self, filename, q, r, testdata):

        """ Append pose estimation for test image to submission. """

        self._append(filename, q, r, testdata, real=False)
        return

    def append_real_test(self, filename, q, r, testdata):

        """ Append pose estimation for real image to submission. """

        self._append(filename, q, r, testdata, real=True)
        return

    def export(self, out_dir='', suffix=None):

        """ Exporting submission json file containing the collected pose estimates. """

        sorted_real_test = sorted(self.real_test_results, key=lambda k: k['filename'])
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        if suffix is None:
            suffix = timestamp
        submission_path = os.path.join(out_dir, 'submission_{}.csv'.format(suffix))
        with open(submission_path, 'w') as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            for result in sorted_real_test:
                csv_writer.writerow([result['testdata'], result['filename'], *(result['q'] + result['r'])])

        print('Submission saved to {}.'.format(submission_path))
        return


#----------------------- Visualize Pose ---------------------#
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