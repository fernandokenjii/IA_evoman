from controller import Controller
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# implements controller structure for player
class player_controller(Controller):
    def __init__(self):
        self.scale = MinMaxScaler()
        self.x_train = []

    def control(self, inputs, controller):
        threshold = 0.5
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
        output = controller.model.predict(inputs)
        output=output[0]
        actions = np.where(output>threshold, 1, 0)
        return actions
    def fit_scale(self):
        self.scale.fit(self.x_train)
