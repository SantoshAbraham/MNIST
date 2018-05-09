'''
This file has an implementation of a neural network 
using the Keras framework with Tensorflow back end
'''
import matplotlib.pyplot as plt
import tensorflow 
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras import optimizers
import keras
from keras.datasets import mnist
from keras.utils import np_utils

#from tensorflow.examples.tutorials.mnist import input_data
#data = input_data.read_data_sets("./MNIST/", one_hot=True)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,x_test.shape, y_train.shape,y_test.shape)
# one hot encode outputs
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#

import numpy as np
import tensorflow as tf
#Model Definition and Hyper Parameters
NumberOfCategories = 10
InputPictureDimensions = [28, 28, 1]

'''
x_train = x_train.reshape(x_train.shape[0], InputPictureDimensions[0], InputPictureDimensions[1], InputPictureDimensions[2])
x_test = x_test.reshape(x_test.shape[0], InputPictureDimensions[0], InputPictureDimensions[1], InputPictureDimensions[2])

y_train = keras.utils.to_categorical(y_train, NumberOfCategories)
y_test = keras.utils.to_categorical(y_test, NumberOfCategories)
y_train = y_train.reshape(y_train.shape[0], NumberOfCategories)
y_test = y_test.reshape(y_test.shape[0], NumberOfCategories)
print(x_train.shape,x_test.shape, y_train.shape,y_test.shape)
NetworkArchitecture = {
    'LearningRate' : 0.001,
    'NumberOfLayers': 5,
    'NumberOfCategories': NumberOfCategories,
    'layer_0_type': 'Input',
    'layer_0_InputShape': InputPictureDimensions,   
    #
    'layer_1_type': 'Conv',
    'layer_1_kernel_size': 5,
    'layer_1_NumFilters': 32,
    'layer_1_PoolSize': 2,
    'layer_1_Padding': 'same',
    'layer_1_ConvStride': 1,
    'layer_1_PoolStride': 2,
    'layer_1_Activation': tf.nn.relu,
    #
    'layer_2_type': 'Conv',
    'layer_2_kernel_size': 5,
    'layer_2_NumFilters': 64,
    'layer_2_PoolSize': 2,
    'layer_2_Padding': 'same',
    'layer_2_ConvStride': 1,
    'layer_2_PoolStride': 2,
    'layer_2_Activation': tf.nn.relu,    
    #
    'layer_3_type': 'Dense',
    'layer_3_Activation': tf.nn.relu,
    'layer_3_size': 1024,
    'layer_3_dropout': 0.4,
    #
    'layer_4_type': 'Dense',
    'layer_4_Activation': 'softmax',
    'layer_4_size': NumberOfCategories,
    'layer_4_dropout': 0    
}

'''
x_train = x_train.reshape(x_train.shape[0], InputPictureDimensions[0]* InputPictureDimensions[1]* InputPictureDimensions[2])
x_test = x_test.reshape(x_test.shape[0], InputPictureDimensions[0]* InputPictureDimensions[1] * InputPictureDimensions[2])

y_train = keras.utils.to_categorical(y_train, NumberOfCategories)
y_test = keras.utils.to_categorical(y_test, NumberOfCategories)
y_train = y_train.reshape(y_train.shape[0], NumberOfCategories)
y_test = y_test.reshape(y_test.shape[0], NumberOfCategories)
print(x_train.shape,x_test.shape, y_train.shape,y_test.shape)
NetworkArchitecture = {
    'LearningRate' : 0.01,
    'NumberOfLayers': 5,
    #
    'NumberOfCategories': NumberOfCategories,
    'layer_0_type': 'Input',
    'layer_0_InputShape': InputPictureDimensions,        
    #
    'layer_1_type': 'Dense',
    'layer_1_Activation': 'relu',
    'layer_1_size': 1000,
    'layer_1_dropout': 0.2, 
    #
    'layer_2_type': 'Dense',
    'layer_2_Activation': 'relu',
    'layer_2_size': 1000,
    'layer_2_dropout': 0.2,
    #
    'layer_3_type': 'Dense',
    'layer_3_Activation': 'relu',
    'layer_3_size': 1000,
    'layer_3_dropout': 0.2,
    #
    'layer_4_type': 'Dense',
    'layer_4_Activation': 'softmax',
    'layer_4_size': NumberOfCategories,
    'layer_4_dropout': 0    
}

#Initialize input and Output PlaceHolders


NumLayers = NetworkArchitecture['NumberOfLayers']
model = Sequential()
for i in range(1,NumLayers):
    PreFix = 'layer_'+str(i)+'_'
    PrevPreFix = 'layer_'+str(i-1)+'_'
    
    #Convolutional Layer
    if NetworkArchitecture[PreFix+'type'] == 'Conv':
        if i==1:
            PictShape = NetworkArchitecture[PrevPreFix + 'InputShape'] 
            input_shape=(PictShape[0], PictShape[1], PictShape[2])
            
            model.add(
                Conv2D(
                NetworkArchitecture[PreFix+'NumFilters'],
                kernel_size = NetworkArchitecture[PreFix+'kernel_size'],
                activation = NetworkArchitecture[PreFix+'Activation'],
                strides = NetworkArchitecture[PreFix+'ConvStride'],
                input_shape = input_shape
                )
            )
        else:
            model.add(
                Conv2D(
                NetworkArchitecture[PreFix+'NumFilters'],
                kernel_size = NetworkArchitecture[PreFix+'kernel_size'],
                activation = NetworkArchitecture[PreFix+'Activation'],
                strides = NetworkArchitecture[PreFix+'ConvStride']             
                )
            )
        model.add(
             MaxPooling2D(pool_size=NetworkArchitecture[PreFix+'PoolSize'],             
                          strides=NetworkArchitecture[PreFix+'PoolStride'], 
                          padding=NetworkArchitecture[PreFix+'Padding']
                        )
        )
  
    if NetworkArchitecture[PreFix+'type'] == 'Dense':
        if NetworkArchitecture[PrevPreFix+'type'] == 'Conv':
            model.add(Flatten())
        if i == 1: #First layer
            input_shape = NetworkArchitecture[PrevPreFix + 'InputShape'][0]* \
            NetworkArchitecture[PrevPreFix + 'InputShape'][1] * \
            NetworkArchitecture[PrevPreFix + 'InputShape'][2]      
            print('InputShape',input_shape, NetworkArchitecture[PrevPreFix + 'InputShape'])
            model.add(Dense(
                units = NetworkArchitecture[PreFix+'size'],
                activation=NetworkArchitecture[PreFix+'Activation'],
                input_shape = [input_shape , ]
                )
            )
        else:
            model.add(Dense(
                units = NetworkArchitecture[PreFix+'size'],
                activation=NetworkArchitecture[PreFix+'Activation']             
                )
            )
            
        if NetworkArchitecture[PreFix+'dropout'] > 0:
            model.add(Dropout(NetworkArchitecture[PreFix+'dropout']))

optimizer = optimizers.Adam(lr = NetworkArchitecture['LearningRate'])
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

'''
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
batch_size = 100
epochs = 500
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

