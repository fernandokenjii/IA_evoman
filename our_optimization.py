# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from our_controller import player_controller

# imports other libs
import time
import numpy as np
import pandas as pd
from scipy.stats import hmean
import matplotlib.pyplot as plt
import pickle
import glob, os
import math

mode = 'test'

parameters = {
    'enemies' : (1,4,6,7),
    'timeexpire' : 600,
    'number_of_iterations' : 150,
    'population_size' : 10,
    'generated_on_mutation' : 5,
    'mutation_alpha' : 0.5,              # using after doomsday and crossover
    'doomsday_interval' : 20,
    'doomsday_survivals' : 5,
    'neuronet_inicialization' : (-1,1),
    'gamma' : 0.7,
    'layers' : [
        {'units':32, 'activation':'sigmoid', 'input_dim':14},
        {'units':12, 'activation':'sigmoid'},
        {'units':5, 'activation':'sigmoid'} #output
    ],
    'number_of_projectiles' : 5
}

best_agents = {
    'first' : [],
    'second' : [],
    'third' : [],
    'agent' : []
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
        self.results = None
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
            F = [muta(nn, 1-(alpha_muta*it)) for nn in P]
            G = [muta(nn, parameters['mutation_alpha']) for nn in crossover(P, f_num)]
            P = F + G
            P = select(P, n_pop)
            best_agents['first'].append(test_agent(P[0]))
            best_agents['agent'].append(P[0])
            best_agents['second'].append(test_agent(P[1]))
            best_agents['third'].append(test_agent(P[2]))
            if it%parameters['doomsday_interval'] == 0 and it != 0:
                P = P[:parameters['doomsday_survivals']]
                N = [NeuroNet() for _ in range(f_num-parameters['doomsday_survivals'])]
                evaluate(N)
                F = [muta(nn, parameters['mutation_alpha']) for nn in N]
                P += F
            pickle.dump([it+1, P, best_agents], open(experiment_name+'/Evoman.pkl', 'wb'))
    # os.remove('Evoman.pkl')
    env.update_parameter('speed', "normal")
    env.update_parameter('timeexpire', 3000)
    df = pd.DataFrame(best_agents['first'])
    plt.plot(df['result'], label='mean')
    plt.plot(df['fitness'], label='fitness')
    plt.legend()
    plt.savefig(experiment_name+'/results.png')
    plt.close('all')
    best = best_agents['agent'][df['result'].idxmax()]
    df.to_csv(experiment_name+'/results.csv')
    for en in enemies:
        env.update_parameter('enemies', [en])
        simulation(env, best)
    others = [en for en in range(1, 9) if en not in enemies]
    for en in others:
        env.update_parameter('enemies', [en])
        simulation(env, best)
    return P

def test_agent(agent):            # use after select function only
    if agent.results is not None:
        return agent.results
    results = {}
    avarage_helper = []
    gains = []
    env.update_parameter('timeexpire', 3000)
    for en in enemies:
        env.update_parameter('enemies', [en])
        f, p, e, t = simulation(env, agent)
        avarage_helper.append([p, e])
        results[en] = [p, e]
        gains.append(100.01 + p - e)
    results['avarage_train'] = np.mean(avarage_helper, axis=0)
    avarage_helper = []
    others = [en for en in range(1, 9) if en not in enemies]
    for en in others:
        env.update_parameter('enemies', [en])
        f, p, e, t = simulation(env, agent)
        avarage_helper.append([p, e])
        results[en] = [p, e]
        gains.append(100.01 + p - e)
    results['avarage_test'] = np.mean(avarage_helper, axis=0)
    results['avarage'] = np.mean((results['avarage_train'], results['avarage_test']), axis=0)
    results['result'] = hmean(gains)
    results['fitness'] = agent.fitness
    agent.results = results
    env.update_parameter('timeexpire', parameters['timeexpire'])
    return results


def start_or_load(n_iter, n_pop):
    if os.path.exists(experiment_name+'/Evoman.pkl'):
        a = pickle.load(open(experiment_name+'/Evoman.pkl', 'rb'))
        if a[0] < n_iter or mode.lower() == 'test':
            global best_agents
            best_agents = a[2]
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
    F.insert(0, nn)
    return select(F, 1)[0]

def select(P, n):
    P.sort(key=lambda nn: nn.fitness, reverse=True) # sort from bigger to lower
    return P[:n]

ini = time.time()  # sets time marker

# runs simulation
def simulation(env,y):
    player_controller.set_weights(y.get_weights())
    _ ,p,e,t = env.play(pcont=y) #fitness, playerlife, enemylife, gametime
    f = parameters['gamma'] * (100-e) + (1-parameters['gamma']) * p - math.log(t)
    return f, p, e, t

# evaluation
def evaluate(x):
    fitness=[]
    for en in enemies:
        env.update_parameter('enemies', [en])
        fitness.append((list(map(lambda y: simulation(env,y)[0], x))))
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


