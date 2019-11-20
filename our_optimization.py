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
    speed="fastest",
    timeexpire=600
)

enemies = (1,3,6,7)

class NeuroNet:
    def __init__(self, weights=None):
        model = Sequential()
        model.add(Dense(8, kernel_initializer=keras_init.RandomUniform(minval=-.5, maxval=.5), activation='tanh', input_dim=20)) # TODO: check right activation function
        #model.add(Dense(6, kernel_initializer=keras_init.RandomUniform(minval=-0.5, maxval=0.5), activation='tanh')) # TODO: check right activation function
        model.add(Dense(5, activation='sigmoid')) # output
        if (weights != None):
            model.set_weights(weights)
        self.model = model
        self.fitness = -math.inf

    def get_weights(self):
        return self.model.get_weights()

def GA(n_iter, n_pop):
    f_num = n_pop
    start, P = start_or_load(n_iter, n_pop)
    if start == 0:
        evaluate(P)
    for it in range(start, n_iter):
        print(it)
        print(P[0].fitness)
        F = [muta(nn) for nn in crossover2(P, f_num)]
        evaluate(F)
        F += [muta(nn) for nn in P]
        P = F
        P = select(P, n_pop)
        if it%10 == 0 and it != 0:
            P = P[:3]
            N = [NeuroNet() for _ in range(f_num-3)]
            evaluate(N)
            P += N
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
    F = []
    for _ in range(5):
        f = []
        for layer in weights:
            l=[]
            shape = layer.shape
            for gene in np.nditer(layer):
                l.append(gene + np.random.normal(0, 0.5))
            l = np.array(l).reshape(shape)
            f.append(l)
        F.append( NeuroNet(f) )
    evaluate(F)
    F.append(nn)
    return select(F, 1)[0]

def select(P, n):
    P.sort(key=lambda nn: nn.fitness, reverse=True) # sort from bigger to lower
    return P[:n]

ini = time.time()  # sets time marker

# runs simulation
def simulation(env,y):
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

GA(60,10)

fim = time.time() # prints total execution time for experiment

env.state_to_log() # checks environment state
