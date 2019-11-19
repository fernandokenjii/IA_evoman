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
