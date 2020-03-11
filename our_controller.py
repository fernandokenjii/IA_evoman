from controller import Controller
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
import keras.initializers as keras_init

def norm(v):
    if (v == 0):
        return 0
    sinal = 1 if (v > 0) else -1
    return sinal * 2 ** (-(v / 150)**2)

# implements controller structure for player
class player_controller(Controller):
    def __init__(self, parameters):
        self.parameters = parameters
        model = Sequential()
        model.add(Dense(parameters['layer1_shape'], activation=parameters['layer_activation'], input_dim=14))
        model.add(Dense(parameters['layer2_shape'], activation=parameters['layer_activation']))
        model.add(Dense(5, activation='sigmoid')) # output
        self.model = model
        weights = model.get_weights()
        self.shapes = []
        for weight in weights:
            self.shapes.append(weight.shape)

    def control(self, inputs, controller):
        threshold = 0.5
        # turn player's actual direction into 'is player looking at opponent?'
        inputs[2] = 0 if (inputs[0]<0) ^ (inputs[2]>0) else 1
        inputs[0] = norm(inputs[0])
        inputs[1] = norm(inputs[1])

        projectile = inputs[4:].reshape((-1,2))
        dist = np.sqrt(np.sum((projectile) ** 2, axis=1)).reshape((-1,1))
        a = np.hstack((projectile, dist)) # Concatenate each position with its distance
        a = a[a[:,2].argsort()] # Order rows by the third column (distance)
        a = a[:,:2] # Remove distances from array
        a = a[:self.parameters['number_of_projectiles'],].flatten()
        a = list(map(norm, a))
        inputs = np.concatenate((inputs[:4], a))

        if controller is 'None':
            self.x_train.append(inputs)
            return [np.random.choice([1,0]) for _ in range(5)]

        inputs = np.reshape(inputs, (-1, 14))
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

