import os
from glob import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Flatten
from keras import backend as K


def activation(y):
    return keras.backend.switch(y > 0.5, K.maximum(y, 1), K.minimum(y, 0))


def CNN(in_shape, out_shape):
    X_input = tf.keras.Input(in_shape)

    X = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv0')(X_input)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn0')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1')(X)
    X = tf.keras.layers.BatchNormalization(axis=-1, name='bn1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    # X = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2')(X)
    # X = tf.keras.layers.BatchNormalization(axis=-1, name='bn2')(X)
    # X = tf.keras.layers.Activation('relu')(X)
    # X = tf.keras.layers.Dropout(rate=0.2)(X)

    X = tf.keras.layers.MaxPool2D(pool_size=2)(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(out_shape[0]*out_shape[1], activation='sigmoid', name='fc')(X)
    X = tf.keras.layers.Reshape(out_shape)(X)
    X = tf.keras.layers.Lambda(activation)(X)

    m = tf.keras.Model(inputs=X_input, outputs=X)

    return m


os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder = "data/training_dataset/processed_images/"
mask_folder = "data/training_dataset/masks/"

images_path = glob(folder + "*.png")
masks_path = glob(mask_folder + "*.png")
dataset = []
masks = []

folder = "data/training_dataset/processed_images/"
mask_folder = "data/training_dataset/masks/"

images_path = sorted(glob(folder + "*.png"))
masks_path = sorted(glob(mask_folder + "*.png"))

dataset = []
masks = []

tot = 3220
for path in images_path:
    img = cv.imread(path, cv.CV_8UC4)
    _, c2, c3, c4 = cv.split(img)
    # c1, c2, c3, _ = cv.split(img)
    img = cv.merge((c4, c2, c3))
    # img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    img = img / 255
    img_resized = cv.resize(img, (224, 224))
    dataset.append(img_resized)

dataset = np.array(dataset)

for path in masks_path:
    mask = cv.imread(path, cv.CV_8UC1)
    mask = mask / 255
    # mask_resized = mask.reshape(0, 1)
    mask_resized = cv.resize(mask, (224, 224))
    np_mask = np.asarray(mask_resized).reshape(-1)
    masks.append(np_mask)

masks = np.array(masks)

input_shape = dataset[0].shape
output_shape = masks[0].shape

print('Input shape: ', input_shape)
print('Output shape: ', output_shape)
print('Number of images: ', len(dataset))

train_dataset, val_dataset, train_labels, val_labels = train_test_split(dataset, masks, train_size=0.7, random_state=1)
val_dataset, test_dataset, val_labels, test_labels = train_test_split(val_dataset, val_labels, train_size=0.7, random_state=1)

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
batch_size = 64
num_epochs = 10

# print("CNN ----------------")
# model = CNN(input_shape, output_shape)

# print("VGG ----------------")
# vgg = tf.keras.applications.VGG19(input_shape=(224, 224, 3), include_top=True, weights=None, classes=50176, classifier_activation='sigmoid')
# x = Flatten()(vgg.output)
# x = tf.keras.layers.Dense(output_shape[0]*output_shape[1], activation='sigmoid', name='fc')(x)
# x = tf.keras.layers.Reshape(output_shape)(x)
# x = tf.keras.layers.Lambda(activation)(x)
# model = Model(vgg.input, x)

# print("Inception ----------------")
# vgg = tf.keras.applications.InceptionResNetV2(input_shape=input_shape, include_top=False)
# x = Flatten()(vgg.output)
# x = tf.keras.layers.Dense(output_shape[0]*output_shape[1], activation='sigmoid', name='fc')(x)
# x = tf.keras.layers.Reshape(output_shape)(x)
# x = tf.keras.layers.Lambda(activation)(x)
# model = Model(vgg.input, x)

print("MobileNet ----------------")
mobnet = tf.keras.applications.MobileNet(input_shape=input_shape, weights=None, include_top=True, classes=50176, classifier_activation='sigmoid')

mobnet.compile(optimizer="adam", loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
mobnet.summary()

history = mobnet.fit(x=train_dataset, y=train_labels, validation_data=(val_dataset, val_labels), epochs=num_epochs,
                    callbacks=[early_stop_callback])
#
# model.save('data/models/model.h5')

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

path = "model/"

full_model = tf.function(lambda inputs: mobnet(inputs))
full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in mobnet.inputs])

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=path+"frozen_models", name="boat.pb", as_text=False)
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=path+"frozen_models", name="boat.pbtxt", as_text=True)

# pred = model.predict(dataset[9:10])
# plt.imshow(pred[0])
# plt.show()
#
# print(masks[9])
# plt.imshow(masks[9])
# plt.show()
