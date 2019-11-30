from controller import Controller
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
import keras.initializers as keras_init


# implements controller structure for player
class player_controller(Controller):
    def __init__(self):
        self.scale = MinMaxScaler((-1,1))
        self.x_train = []
        model = Sequential()
        model.add(Dense(20, activation='tanh', input_dim=10)) # TODO: check right activation function
        model.add(Dense(10, activation='tanh')) # TODO: check right activation function
        model.add(Dense(5, activation='sigmoid')) # output
        self.model = model
        weights = model.get_weights()
        self.shapes = []
        for weight in weights:
            self.shapes.append(weight.shape)

    def control(self, inputs, controller):
        threshold = 0.5
        # turn player's actual direction into 'is player looking at opponent?'
        # inputs[2] = 0 if (inputs[0]<0) ^ (inputs[2]>0) else 1

        projectile = inputs[4:].reshape((-1,2))
        dist = np.sqrt(np.sum((projectile) ** 2, axis=1)).reshape((-1,1))
        a = np.hstack((projectile, dist)) # Concatenate each position with its distance
        a = a[a[:,2].argsort()] # Order rows by the third column (distance)
        a = a[:,:2] # Remove distances from array
        a = a[:3,].flatten()
        inputs = np.concatenate((inputs[:4], a))

        if controller is 'None':
            self.x_train.append(inputs)
            return [np.random.choice([1,0]) for _ in range(5)]

        inputs = self.scale.transform([inputs])
        output = self.model.predict(inputs)
        output=output[0]
        actions = np.where(output>threshold, 1, 0)
        return actions

    def fit_scale(self):
        self.scale.fit(self.x_train)

    def get_shapes(self):
        return self.shapes

    def set_weights(self, weights):
        self.model.set_weights(weights)

