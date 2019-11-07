# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from our_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os


from keras.models import Sequential
from keras.layers import Dense

experiment_name = 'our_tests'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                    enemies=[1],
                    playermode="ai",
                    player_controller=player_controller(),
                    enemymode="static",
                    level=2,
                    speed="fastest")

enemies = (1,8,3,4)

class NeuroNet:
    def __init__(self, weights=None):
    # def __init__(self, _n_hidden):
        # Number of hidden neurons
        # self.n_hidden = [_n_hidden]
        model = Sequential()
        model.add(Dense(5, activation='relu', input_dim=20)) # TODO: check right activation function
        #model.add(Dense(14, activation='relu'))               # TODO: check right activation function
        model.add(Dense(5, activation='sigmoid')) # output
        # model.add(Dense(5, bias_initializer='ones',kernel_initializer='random_uniform', activation='sigmoid'))
        # model.set_weights(weights)
        # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        # model.fit(data,labels,epochs=10,batch_size=32)
        # predictions = model.predict(data)
        if (weights != None):
            model.set_weights(weights)
        # print(model.get_weights())
        # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model



# default environment fitness is assumed for experiment

#env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params



# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x) #fitness, playerlife, enemylife, gametime
    return f


# evaluation
def evaluate(x):
    fitness=[0]* len(enemies)
    for i, en in enumerate(enemies):
        env.update_parameter('enemies', [en])
        fitness[i] = np.array(list(map(lambda y: simulation(env,y), x)))
    return fitness
    

nn = NeuroNet()
print(evaluate([nn]))

fim = time.time() # prints total execution time for experiment

env.state_to_log() # checks environment state
