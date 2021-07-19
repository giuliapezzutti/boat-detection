import os
from glob import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import backend as K


def CNN(in_shape):
    X_input = tf.keras.Input(in_shape)

    X = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1')(X_input)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.1)(X)

    X = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn2')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn3')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid',
                               name='conv4')(X)
    m = tf.keras.Model(inputs=X_input, outputs=X)

    return m


folder = "../data/training_dataset/processed_images/"
mask_folder = "../data/training_dataset/masks/"

images_path = sorted(glob(folder + "*.png"))
masks_path = sorted(glob(mask_folder + "*.png"))

dataset = []
masks = []

for path in images_path:
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    img = img / 255
    dataset.append(img)

dataset = np.array(dataset)

for path in masks_path:
    mask = cv.imread(path, cv.IMREAD_UNCHANGED)
    mask = mask / 255
    mask_resized = np.expand_dims(np.asarray(mask), -1)
    masks.append(mask_resized)

masks = np.array(masks)

input_shape = dataset[0].shape
output_shape = masks[0].shape

print('Input shape: ', input_shape)
print('Output shape: ', output_shape)
print('Number of images: ', len(dataset))

train_dataset, val_dataset, train_labels, val_labels = train_test_split(dataset, masks, train_size=0.7, random_state=1)

early_stop_callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                       keras.callbacks.ModelCheckpoint("../models/model-checkpoint.h5", save_best_only=True)]
batch_size = 64
num_epochs = 20

print("CNN ----------------")
model = CNN(input_shape)

model.compile(optimizer="adam", loss='mse', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
model.summary()

history = model.fit(x=train_dataset, y=train_labels, validation_data=(val_dataset, val_labels), epochs=num_epochs,
                    callbacks=[early_stop_callback])

model.save('../models/model.h5')

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function(
    [tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="../models/", name="model.pb", as_text=False)
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=path+"frozen_models", name="boat.pbtxt", as_text=True)