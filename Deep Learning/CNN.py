# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:04:37 2020

@author: kevin
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255, #applies feature scaling to the images
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) #these features perform image augmentation to prevent overfitting by diversifying the image orientation, rotation and other features
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), #changes the final size of image that will be fed to the cnn for processing
        batch_size=32,
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255) #test set images are feature scaled but not edited to prevent information leakage
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64), #changes the final size of image that will be fed to the cnn for processing
        batch_size=32,
        class_mode='binary')

cnn = tf.keras.models.Sequential()

#adding convolutional layers
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3))) #input shape: 1st 2 args is input image final size, 3rd is how many color channels there are

#adding pooling layers
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

#adding second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')) #input shape not needed as it is only for the first layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

#flattening
cnn.add(tf.keras.layers.Flatten())

#connecting to ANN
cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) #much higher no. of units since computer vision is much more complex
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #loss function must be this one for binary outcomes, categorical is 'categorical_crossentropy'
cnn.fit(x = training_set, validation_data = test_set, epochs=25) #training the cnn model on the dataset


#predicting individual images
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64)) #image is turned into PIL format
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0) #adds a dimension to the image array to fit the format for the cnn
result = cnn.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)