# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from our_controller import player_controller

# imports other libs
import time
import numpy as np
import pickle
import glob, os
import math

mode = 'test'

parameters = {
    'enemies' : (1,3,6,7),
    'timeexpire' : 600,
    'number_of_iterations' : 150,
    'population_size' : 10,
    'generated_on_mutation' : 5,
    'mutation_alpha' : 0.5,              # using after doomsday and crossover
    'doomsday_interval' : 20,
    'doomsday_survivals' : 5,
    'neuronet_inicialization' : (-1,1),
    'layer1_shape' : 32,
    'layer2_shape' : 12,
    'layer_activation' : 'relu',
    'number_of_projectiles' : 5
}

experiment_name = 'our_tests'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

player_controller = player_controller(parameters)

if mode.lower() != 'test':
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(
    experiment_name=experiment_name,
    enemies=[1],
    playermode="ai",
    player_controller=player_controller,
    enemymode="static",
    level=2,
    speed="fastest",
    timeexpire=parameters["timeexpire"]
)

enemies = parameters['enemies']

class NeuroNet:
    def __init__(self, weights=None):
        self.weights = []
        if (weights is not None):
            self.weights = weights
        else:
            for shape in player_controller.get_shapes():
                self.weights.append(np.random.uniform(*parameters['neuronet_inicialization'], shape))
        self.fitness = -math.inf

    def get_weights(self):
        return self.weights

def GA(n_iter, n_pop):
    f_num = n_pop
    start, P = start_or_load(n_iter, n_pop)
    alpha_muta = 1/n_iter
    if start == 0:
        evaluate(P)
    if mode.lower() != 'test':
        for it in range(start, n_iter):
            log_str = f'GENERATION: {it} | BEST FITNESS: {P[0].fitness}'
            print(log_str)
            log_to_file(log_str)
            F = [muta(nn, parameters['mutation_alpha']) for nn in crossover(P, f_num)]
            F += [muta(nn, 1-(alpha_muta*it)) for nn in P]
            P = F
            P = select(P, n_pop)
            if it%parameters['doomsday_interval'] == 0 and it != 0:
                P = P[:parameters['doomsday_survivals']]
                N = [NeuroNet() for _ in range(f_num-parameters['doomsday_survivals'])]
                evaluate(N)
                F = [muta(nn, parameters['mutation_alpha']) for nn in N]
                P += F
            pickle.dump([it+1, P], open(experiment_name+'/Evoman.pkl', 'wb'))
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
        if a[0] < n_iter or mode.lower() == 'test':
            return a[0], a[1]
    return 0, [NeuroNet() for _ in range(n_pop)]

def calc_weights(nn, alpha):
    weights = nn.get_weights()
    new_weights = [(weight * alpha) for weight in weights]
    return new_weights

def crossover(P, n):
    F = []
    pairs = np.random.choice(P, (n//2, 2), False)
    for pair in pairs:
        a = np.random.random()
        w1 = calc_weights(pair[0], a)
        w2 = calc_weights(pair[1], 1 - a)
        w = [(w1[j] + w2[j]) for j in range(len(w1))]
        F.append( NeuroNet(w) )
    evaluate(F)
    return F

def muta(nn, alpha):
    weights = nn.get_weights()
    F = []
    for _ in range(parameters['generated_on_mutation']):
        f = []
        for layer in weights:
            l=[]
            shape = layer.shape
            for gene in np.nditer(layer):
                l.append(gene + np.random.normal(0, alpha))
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
    player_controller.set_weights(y.get_weights())
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

def log_to_file(str):
    file = open(experiment_name+'/results.txt', 'a')
    file.write(str + "\n")
    file.close()

GA(parameters['number_of_iterations'], parameters['population_size'])


