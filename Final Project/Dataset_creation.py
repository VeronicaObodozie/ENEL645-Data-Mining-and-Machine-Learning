# Import important modules
import os
import tensorflow as tf
from tensorflow import keras

import numpy as np
from sklearn.metrics import cohen_kappa_score

# Load dataset
def get_images_from_path(dataset_path):
    """ Get images from path and normalize them applying channel-level normalization. """

    # loading all images in one large batch
    tf_eval_data = tf.keras.utils.image_dataset_from_directory(dataset_path, image_size=input_shape[:2], shuffle=False, 
                                                               batch_size=100000)

    # extract images and targets
    for tf_eval_images, tf_eval_targets in tf_eval_data:
        break

    return tf.convert_to_tensor(tf_eval_images), tf_eval_targets

# Breakdown images into tiles


# Early stopping during training

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

# Metrics
def metric_eval(model, x_test, y_test):
     #Model shall be compiled before testing.
    model.compile()

    #Creating empty predictions
    predictions = np.zeros(len(y_train), dtype=np.int8)

    # inference loop
    for e, (image, target) in enumerate(zip(x_train, y_train)):
        image = np.expand_dims(np.array(image), axis=0)
        output = model.predict(image)
        predictions[e] = np.squeeze(output).argmax()

    #Keras model score
    score_keras = 1 - cohen_kappa_score(y_train.numpy(), predictions)
    print("Score:",score_keras)