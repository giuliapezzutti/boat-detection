import os
from glob import glob
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten, Dense
from keras import backend as K


@tf.function
def customed_activation(y):
    if K.greater(y, 0.5):
        return 1
    else:
        return 0


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

    m = tf.keras.Model(inputs=X_input, outputs=X)

    return m


os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder = "data/training_dataset/processed_images/"
mask_folder = "data/training_dataset/masks/"

images_path = glob(folder + "*.png")
masks_path = glob(mask_folder + "*.png")
dataset = []
masks = []

for label_path in images_path:
    img = cv.imread(label_path, cv.CV_8UC4)
    _, c2, c3, c4 = cv.split(img)
    img = cv.merge((c2, c3, c4))
    img_resized = cv.resize(img, (32, 64))
    dataset.append(img_resized)
dataset = np.array(dataset)

for label_path in masks_path:
    mask = cv.imread(label_path, cv.CV_8UC1)
    mask_resized = cv.resize(mask, (32, 64))
    for i in range(mask_resized.shape[0]):
        for j in range(mask_resized.shape[1]):
            mask_resized[i, j] = mask_resized[i, j]/255
    masks.append(mask_resized)
masks = np.array(masks)

input_shape = dataset[0].shape
output_shape = masks[0].shape

print('Input shape: ', input_shape)
print('Output shape: ', output_shape)

train_dataset, val_dataset, train_labels, val_labels = train_test_split(dataset, masks, train_size=0.7, random_state=1)
val_dataset, test_dataset, val_labels, test_labels = train_test_split(val_dataset, val_labels, train_size=0.7, random_state=1)

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
batch_size = 64
num_epochs = 20

# print("CNN ----------------")
# model = CNN(input_shape, output_shape)
# model.compile(optimizer="adam", loss='mse', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
#
# history = model.fit(x=train_dataset, y=train_labels, validation_data=(val_dataset, val_labels), epochs=num_epochs,
#                     callbacks=[early_stop_callback])
#
# model.save('data/models/CNN.h5')

vgg = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
x = Flatten()(vgg.output)
x = tf.keras.layers.Dense(output_shape[0]*output_shape[1], activation='sigmoid', name='fc')(x)
x = tf.keras.layers.Reshape(output_shape)(x)
x = tf.keras.layers.Activation(customed_activation)(x)
model = Model(vgg.input, x)
print(vgg.summary())
print(model.summary())

model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

history = model.fit(x=train_dataset, y=train_labels, validation_data=(val_dataset, val_labels), epochs=num_epochs,
                    callbacks=[early_stop_callback])

model.save('data/models/VGG16.h5')

pred = model.predict(dataset[0:1])
print(pred)
print(pred.shape)
plt.imshow(pred[0])
plt.show()

plt.imshow(masks[0])
plt.show()
