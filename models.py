import json
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam

class OneNeuronNN(Sequential):

    def __init__(self, inputDimensions, learningRate=0.005, activation="linear"):

        self = Sequential()

        self.add(Dense(units=4, input_dim=inputDimensions, activation=activation, init='uniform'))
        optimizer = SGD(lr=learningRate)
        self.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])


class CNN(Sequential):

    # inputDimensions already set as rows cols and channels
    # def __init__(self, inputDimensions, learningRate=0.005, activation="linear"):
    def __init__(self, learningRate=0.005, inputDimensions = (18, 34, 1)):

        # img_channels = 1 #We stack 4 frames (Only 1 frame is necesary?)

        print("Now we build the model")
        self = Sequential()
        
        self.add(Conv2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=inputDimensions))  #80*80*4 for Tensorflow
        self.add(Activation('relu'))
        self.add(Conv2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        self.add(Activation('relu'))
        self.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        self.add(Activation('relu'))
        self.add(Flatten())
        self.add(Dense(512))
        self.add(Activation('relu'))
        self.add(Dense(2))
    
        adam = Adam(lr=learningRate)
        self.compile(loss='mse',optimizer=adam)
        print("We finish building the model")

def createCNN(learningRate = 0.005, inputDimensions = (18, 34, 1)):
    
    model = Sequential()

    model.add(Conv2D(32, (8, 8), padding="same", strides=(4, 4), input_shape=inputDimensions))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), padding="same", strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1) ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(5))

    adam = Adam(lr=learningRate)
    model.compile(loss='mse',optimizer=adam)

    return model


# def createCNN(learningRate = 0.005, inputDimensions = (18, 34, 1)):
    
    # model = Sequential()

    # model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', input_shape=inputDimensions))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dense(5))

    # adam = Adam(lr=learningRate)
    # model.compile(loss='mse',optimizer=adam)

    # return model