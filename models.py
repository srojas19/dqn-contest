import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

class OneNeuronNN(Sequential):

    def __init__(self, inputDimensions, learningRate=0.005, activation="linear"):

        Sequential.__init__()

        self.add(Dense(units=4, input_dim=inputDimensions, activation=activation, init='uniform'))
        optimizer = SGD(lr=learningRate)
        self.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])


class CNN(Sequential):

    def __init__(self, inputDimensions, learningRate=0.005, activation="linear"):

        img_rows , img_cols = 80, 80
        #Convert image into Black and white
        img_channels = 4 #We stack 4 frames

        print("Now we build the model")
        Sequential.__init__()
        
        self.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
        self.add(Activation('relu'))
        self.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        self.add(Activation('relu'))
        self.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        self.add(Activation('relu'))
        self.add(Flatten())
        self.add(Dense(512))
        self.add(Activation('relu'))
        self.add(Dense(2))
    
        adam = Adam(lr=learningRate)
        self.compile(loss='mse',optimizer=adam)
        print("We finish building the model")