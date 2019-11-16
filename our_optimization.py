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
import keras.initializers as keras_init

experiment_name = 'our_tests'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

player_controller = player_controller()

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(
    experiment_name=experiment_name,
    enemies=[1],
    playermode="ai",
    player_controller=player_controller,
    enemymode="static",
    level=2,
    speed="fastest"
)

enemies = (3,4,6,7)

class NeuroNet:
    def __init__(self, weights=None):
        model = Sequential()
        model.add(Dense(8, kernel_initializer=keras_init.RandomUniform(minval=-1., maxval=1.), activation='tanh', input_dim=20)) # TODO: check right activation function
        #model.add(Dense(6, kernel_initializer=keras_init.RandomUniform(minval=-0.5, maxval=0.5), activation='tanh')) # TODO: check right activation function
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
    f_num = 20
    for it in range(start, n_iter):
        print(it)
        print(P[0].fitness)
        Psel = select(P, f_num)
        F = [muta(nn) for nn in crossover2(Psel, f_num)]
        evaluate(F)
        P = P + F
        P = select(P, n_pop)
        pickle.dump([it+1, P, player_controller.scale], open(experiment_name+'/Evoman.pkl', 'wb'))
    # os.remove('Evoman.pkl')
    env.update_parameter('speed', "normal")
    for en in enemies:
        env.update_parameter('enemies', [en])
        simulation(env, P[0])
    others = [en for en in range(1, 9) if en not in enemies]
    for en in others:
        env.update_parameter('enemies', [en])
        simulation(env, P[0])
    return P

def start_or_load(n_iter, n_pop):
    if os.path.exists(experiment_name+'/Evoman.pkl'):
        a = pickle.load(open(experiment_name+'/Evoman.pkl', 'rb'))
        if a[0] < n_iter:
            player_controller.scale = a[2]
            return a[0], a[1]
    fit_scale()
    return 0, [NeuroNet() for _ in range(n_pop)]

def calc_weights(nn, alpha):
    weights = nn.get_weights()
    new_weights = [(weight * alpha) for weight in weights]
    return new_weights

def crossover(P, n):
    F=[]
    weight1 = calc_weights(P[0], 0.5)
    for i in range(1, n):
        weight2 = calc_weights(P[i], 0.5)
        weight = [(weight1[j] + weight2[j]) for j in range(len(weight1))]
        F.append( NeuroNet(weight) )
    return F

def crossover2(P, n):
    F = []
    pairs = np.random.choice(P, (n//2, 2), False)
    for pair in pairs:
        a = np.random.random()
        w1 = calc_weights(pair[0], a)
        w2 = calc_weights(pair[1], 1 - a)
        w = [(w1[j] + w2[j]) for j in range(len(w1))]
        F.append( NeuroNet(w) )
    return F

def muta(nn):
    weights = nn.get_weights()
    for layer in weights:
        if len(np.shape(layer)) > 1:
            for gene in layer:
                idx = np.random.randint(0, len(gene))
                gene[idx] += np.random.normal(0, 0.1)
        else:
            idx = np.random.randint(0, len(layer))
            layer[idx] += np.random.normal(0, 0.1)
    nn.model.set_weights(weights)
    return nn

def select(P, n):
    P.sort(key=lambda nn: nn.fitness, reverse=True) # sort from bigger to lower
    return P[:n]

ini = time.time()  # sets time marker

# runs simulation
def simulation(env,y):
    #if y.fitness > -math.inf:
    #    return y.fitness
    f,p,e,t = env.play(pcont=y) #fitness, playerlife, enemylife, gametime
    return f

# evaluation
def evaluate(x):
    fitness=[]
    for en in enemies:
        env.update_parameter('enemies', [en])
        fitness.append((list(map(lambda y: simulation(env,y), x))))
    arrray = np.array(fitness)
    fitness = arrray.sum(axis=0)
    fitness /= 8
    fitness += arrray.min(axis=0)
    for i, y in enumerate(x):
        y.fitness = fitness[i]

    return fitness

def fit_scale():
    for en in enemies:
        env.update_parameter('enemies', [en])
        env.play()
    player_controller.fit_scale()

GA(30,20)

fim = time.time() # prints total execution time for experiment

env.state_to_log() # checks environment state
