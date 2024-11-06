# Importing important modules and functions
import os

import tensorflow as tf
from tensorflow import keras

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import re
import glob
import os
import sys

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import AdamW

from tensorflow.keras import optimizers, utils, models
from tensorflow.keras.utils import to_categorical # Function to convert labels to one-hot encoding
import tensorflow_hub as hub
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import applications
from tensorflow.keras.regularizers import l2, l1

from efficientnet_lite import EfficientNetLiteB0

# Device

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)

#-------------------Data Pre-processing-----------------------#
# Large satellite image must be broken into 200x200 tiles

patches = img.data.unfold(0, 3, 3).unfold(1, 224, 224).unfold(3, 2224, 224)
#------------------------------------------#

#-------------------Data Processing-----------------------#
dataset_path="path to your dataset."

#Loading dataset
x_train, y_train=get_images_from_path(dataset_path)

#------------------------------------------#

#-------------------Model Training-----------------------#
# Model architecture is a modified efficent netlite0 given by ESA
PATH ='./ops_sat.h5'
# Loading the model
input_shape = (200, 200, 3)   # input_shape is (height, width, number of channels) for images
num_classes = 8
model = EfficientNetLiteB0(classes=num_classes, weights=None, input_shape=input_shape, classifier_activation=None)
model.summary()
plot_model(model_cnn, to_file='model_cnn.png')

# Training 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Setup callbacks, logs and early stopping condition
checkpoint_path = "stacking_early_fusion/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
cp = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy',save_best_only=True,verbose=1, mode='max')
csv_logger = callbacks.CSVLogger('stacking_early_fusion/stacking_early.log')
es = callbacks.EarlyStopping(patience = 3, restore_best_weights=True)

# Reduce learning rate if no improvement is observed
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.1, patience=1, min_lr=0.00001)

# TRAINING
 # load data, data augmentation, training, overfitting, transfer-learning etc.
history=model.fit(x_train, y_train, epochs=55, verbose=1, batch_size=8)

 #Saving model
model.save_weights('test.h5')
#------------------------------------------#

#-------------------Testing-----------------------#
model = EfficientNetLiteB0(classes=num_classes, weights=None, input_shape=input_shape, classifier_activation=None)
model.load_weights('ops_sat.h5')
metric_eval(model, x_test, y_test)
#------------------------------------------#

#-------------------Metrics-----------------------#

#------------------------------------------#

""" REFERENCES
# semi supervised learning
# satellite imagery, tile creation

Borra, S., Thanki, R., & Dey, N. (2019). Satellite image analysis : clustering and classification. Springer.

#satellite imagery semi-supervised learning

Han, X., Jiang, Z., Liu, Y., Zhao, J., Sun, Q., & Liu, J. (2024). Semi-Supervised Hyperspectral Image Classification Based on Multiscale Spectral-Spatial Graph Attention Network. IEEE Geoscience and Remote Sensing Letters, 21, 1–5. https://doi.org/10.1109/LGRS.2024.3409553

"""