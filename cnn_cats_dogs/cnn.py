#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 09:45:39 2022

@author: xw
"""

"""categorical output results + 2 convolution layer + 1 hidden layer. dependent and 
independent variables should be constructed by directory structure. like use the feature name instead of 
classification_1, classification_2, etc. """

# Building the CNN, Convolutional Neural Network
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

# adding convolution layer, assume 32 kernels and kernel_size is 3*3
# the number of kernels could be different which depend on demand
classifier.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))

# adding max pooling layer, pool_size is 2*2
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# ------------------------------------------------------------------------------------------------------------
# Adding convolutional layers here, before flatten
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# ------------------------------------------------------------------------------------------------------------

# adding flattening, the output will be the input of ANN
classifier.add(Flatten())

# adding full connection, in this stage, ANN is built which includes 1 hidden layer and 1 output layer
# 128 neurons designed in hidden layer
classifier.add(Dense(units=128, activation='relu'))

# ------------------------------------------------------------------------------------------------------------
# Adding hidden layers here
classifier.add(Dense(units = 6, activation = 'relu'))


# ------------------------------------------------------------------------------------------------------------

classifier.add(Dense(units=1, activation='sigmoid'))

# configuration of the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=500,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=62.5)







