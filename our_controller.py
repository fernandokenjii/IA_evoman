from controller import Controller
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# implements controller structure for player
class player_controller(Controller):
    def __init__(self):
        self.scale = MinMaxScaler()
        self.x_train = []

    def control(self, inputs, controller):
        if controller is 'None':
            self.x_train.append(inputs)
            return [np.random.choice([1,0]) for _ in range(5)]
            
        inputs = self.scale.transform([inputs])
        # for x in range (10000000):
        #     pass
        threshold = 0.5
        output = controller.model.predict(inputs)
        output=output[0]
        
        if output[0] > threshold:
            left = 1
        else:
            left = 0

        if output[1] > threshold:
            right = 1
        else:
            right = 0

        if output[2] > threshold:
            jump = 1
        else:
            jump = 0

        if output[3] > threshold:
            # if inputs[0][0] > 0 and inputs[0][2] == -1:
            #     left=1
            #     right=0
            # elif inputs[0][0] < 0 and inputs[0][2] == 1:
            #     left=0
            #     right=1
            shoot = 1
        else:
            shoot = 0

        if output[4] > threshold:
            release = 1
        else:
            release = 0

        return [left, right, jump, shoot, release]
    def fit_scale(self):
        self.scale.fit(self.x_train)