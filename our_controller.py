from controller import Controller
import numpy as np


# implements controller structure for player
class player_controller(Controller):
    def control(self, inputs, controller):
        # # Normalises the input using min-max scaling
        # inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

        threshold = 0.5

        output = controller.model.predict(inputs)

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
            shoot = 1
        else:
            shoot = 0

        if output[4] > threshold:
            release = 1
        else:
            release = 0

        return [left, right, jump, shoot, release]