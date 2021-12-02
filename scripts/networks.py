import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, AvgPool2D, Activation, Flatten, MaxPool2D, Dropout



class CNN_networks():
    def __init__(self, shape):
        super().__init__()
        self.height, self.width, self.channel = (32, 32, 3)

    # we are not using LeNet.
    def LeNet(self):
        model = keras.models.Sequential()

        model.add(Conv2D(filters=6, kernel_size=5, strides=1,
                         input_shape = [self.height, self.width, self.channel],
                         padding = "same",
                         activation='tanh'))
        # shape = (32, 32, 6)
        model.add(AvgPool2D(pool_size=2, strides = 2))
        model.add(Activation('tanh'))
        # shape = (16, 16, 6)

        model.add(Conv2D(filters=16, kernel_size=5, strides=1,
                         activation='tanh'))
        # shape = (12, 12, 16)
        model.add(AvgPool2D(pool_size=2, strides = 2))
        model.add(Activation('tanh'))
        # shape = (6, 6, 16)

        model.add(Conv2D(filters=120, kernel_size = 5, 
                         strides=1, activation= 'tanh'))
        # shape = (2, 2, 120)

        # Flatten the input for fully connected layers
        model.add(Flatten())

        # fully connected layer
        model.add(Dense(units = 84, activation='tanh'))

        # output layer instead of RBF, I used softmax
        model.add(Dense(units=43, activation='softmax'))

        return model


    def CustomNet(self):
        model = keras.models.Sequential()

        model.add(Conv2D(filters=32, kernel_size=7, strides=1,
                         input_shape = (self.height, self.width, self.channel),
                         padding = "same",
                         activation='relu'))
        # shape = (32, 32, 32)
        model.add(Conv2D(filters=64, kernel_size=3, strides=1,
                         padding = "same",
                         activation='relu'))
        # shape = (32, 32, 64)
        model.add(MaxPool2D(pool_size=2, strides = 1))
        # shape = (31, 31, 64)


        # Flatten the input for fully connected layers
        model.add(Flatten())

        # fully connected layer 2
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(0.05))

        # output layer
        model.add(Dense(units=43, activation='softmax'))

        #compile
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model


    def GoogleNet(self):
        pass


