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
import math


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

    def get_weights(self):
        return self.model.get_weights()

def GA(n_iter, n_pop):
    start, P = start_or_load(n_iter, n_pop)
    if start == 0:
        evaluate(P)
    f_num = 10
    for it in range(start, n_iter):
        Psel = select(P, f_num)
        F = crossover(Psel, f_num)
        evaluate(F)
        P = P + F
        P = select(P, n_pop)
        pickle.dump([it+1, P], open(experiment_name+'/Evoman.pkl', 'wb'))
    # os.remove('Evoman.pkl')
        
    return P

def start_or_load(n_iter, n_pop):
    if os.path.exists(experiment_name+'/Evoman.pkl'):
        a = pickle.load(open(experiment_name+'/Evoman.pkl', 'rb'))
        if a[0] < n_iter:
            return a[0], a[1]
    return 0, [NeuroNet() for _ in range(n_pop)]

def calc_weights(nn, alpha):
    weights = nn.get_weights()
    new_weights = [0] * 2
    for i, weight in evaluate(weights):
        new_weights[i] = weight * alpha
    return new_weights

def crossover(P, n):
    F=[]
    weight1 = calc_weights(P[0], 0.8)
    for i in range(1, n):
        weight2 = calc_weights(P[i], 0.2)
        F = F + NeuroNet(weight1 + weight2)
    weight1 = calc_weights(P[1], 0.8)
    weight2 = calc_weights(P[2], 0.2)
    F = F + NeuroNet(weight1 + weight2)
    return F
    
def muta(nn):
    return nn
    
def select(P, n):
    sorted(P, key=lambda nn: nn.fitness, reverse=True) # sort from bigger to lower
    return P[:n+1] # select n elements 

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
    



GA(30,20)

fim = time.time() # prints total execution time for experiment

env.state_to_log() # checks environment state
