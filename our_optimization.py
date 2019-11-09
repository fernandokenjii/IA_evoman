# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from our_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import pickle
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

enemies = (3,4,6,7)

class NeuroNet:
    def __init__(self, weights=None):
        model = Sequential()
        model.add(Dense(5, activation='tanh', input_dim=20)) # TODO: check right activation function
        model.add(Dense(5, activation='sigmoid')) # output

        if (weights != None):
            model.set_weights(weights)

        self.model = model
        self.fitness = -math.inf

def GA(n_iter, n_pop):
    start, P = start_or_load(n_iter, n_pop)
    for it in range(start, n_iter):
        fitness = evaluate(P)
        pickle.dump([it+1, P], open(experiment_name+'/Evoman.pkl', 'wb'))
    # os.remove('Evoman.pkl')
        
    return P

def start_or_load(n_iter, n_pop):
    if os.path.exists(experiment_name+'/Evoman.pkl'):
        a = pickle.load(open(experiment_name+'/Evoman.pkl', 'rb'))
        if a[0] < n_iter:
            return a[0], a[1]
    return 0, [NeuroNet() for _ in range(n_pop)]
    
def muta(nn):
    return nn
    
def seleciona(P, n):
    pass

ini = time.time()  # sets time marker

# runs simulation
def simulation(env,y):
    if y.fitness > -math.inf:
        return y.fitness
    f,p,e,t = env.play(pcont=y) #fitness, playerlife, enemylife, gametime
    return f

# evaluation
def evaluate(x):
    fitness=[0]* len(enemies)
    for i, en in enumerate(enemies):
        env.update_parameter('enemies', [en])
        fitness[i] = (list(map(lambda y: simulation(env,y), x)))
    fitness = np.array(fitness).sum(axis=0)
    for i, y in enumerate(x):
        y.fitness = fitness[i]

    return fitness
    



GA(3,2)

fim = time.time() # prints total execution time for experiment

env.state_to_log() # checks environment state
