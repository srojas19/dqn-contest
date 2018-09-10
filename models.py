from keras.models import Sequential
from keras import layers
from keras import optimizers

class OneNeuronNN(Sequential):

    def __init__(self, inputDimensions, learningRate=0.005, activation="linear"):

        Sequential.__init__()

        self.add(layers.Dense(units=4, input_dim=inputDimensions, activation=activation, init='uniform'))
        optimizer = optimizers.SGD(lr=learningRate)
        self.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])