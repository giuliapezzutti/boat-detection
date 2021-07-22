import os
from glob import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow import keras


def CNN(in_shape):
    X_input = tf.keras.Input(in_shape)

    X = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1')(X_input)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn1')(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn2')(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn3')(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv4')(X)
    X = tf.keras.layers.Activation('sigmoid')(X)
    m = tf.keras.Model(inputs=X_input, outputs=X)

    return m


if __name__ == '__main__':
    
    # TODO: To be set from the command line
    folder = "../data/training_images/"
    mask_folder = "../data/image_masks/"

    # Extract all the file paths inside the two folders
    images_path = sorted(glob(folder + "*.png"))
    masks_path = sorted(glob(mask_folder + "*.png"))

    # Extract the correspondent files - note that for how the two datasets are created, no checks 
    # on the correspondent read file must be done
    dataset = []
    for path in images_path:
        img = cv.imread(path, cv.IMREAD_UNCHANGED)
        img = img / 255
        dataset.append(img)
    dataset = np.array(dataset)

    masks = []
    for path in masks_path:
        mask = cv.imread(path, cv.IMREAD_UNCHANGED)
        mask = mask / 255
        mask_resized = np.expand_dims(np.asarray(mask), -1)
        masks.append(mask_resized)
    masks = np.array(masks)

    # Set the shapes for the training
    input_shape = dataset[0].shape
    output_shape = masks[0].shape
    print('Input shape: ', input_shape)
    print('Output shape: ', output_shape)
    print('Number of images: ', len(dataset))

    # Divide the whole dataseet in training and validation
    train_dataset, val_dataset, train_labels, val_labels = train_test_split(dataset, masks, train_size=0.8, random_state=1)

    # Set some parameters for the training
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                 keras.callbacks.ModelCheckpoint("../models/model-checkpoint.h5", save_best_only=True)]
    batch_size = 64
    num_epochs = 10

    # Creation of the model
    model = CNN(input_shape)
    model.compile(optimizer="adam", loss='mse', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
    model.summary()

    # Fit of the model
    history = model.fit(x=train_dataset, y=train_labels, validation_data=(val_dataset, val_labels), epochs=num_epochs,
                        callbacks=callbacks)
    model.save('../models/model.h5')

    # Save of .pb frozen graph associated with the trained model - needed for the import in OpenCV
    full_model = tf.function(lambda inputs: model(inputs))
    full_model = full_model.get_concrete_function(
        [tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="../models/", name="model.pb", as_text=False)
