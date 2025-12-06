from copy import deepcopy, copy
from torch.autograd import Variable
import pickle
import numpy as np
import torch
import mo_gymnasium
import gymnasium as gym # Import gymnasium for the wrapper
import json
from .nsga2_tools import *

class NormalizedActions(gym.ActionWrapper): # Inherit from gymnasium.ActionWrapper

    def action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    #v = 1. / np.sqrt(fanin)
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)

def to_numpy(var):
    return var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False):
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)

def pickle_obj(filename, object):
    handle = open(filename, "wb")
    pickle.dump(object, handle)

def unpickle_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def min_max_normalize(x):
    min_x = np.min(x)
    max_x = np.max(x)
    return (x - min_x) / (max_x - min_x)

def is_lnorm_key(key):
    return key.startswith('lnorm')

def parse_json(json_file):
    file = open(json_file)
    data = json.load(file)
    file.close()
    return data

from pygmo import hypervolume
def calculate_hv(points, ref_point=np.zeros((2, ))):
    if len(ref_point) != points.shape[-1]:
        ref_point = np.zeros((points.shape[-1], ))
    hv = hypervolume(points)
    return hv.compute(ref_point)

def calculate_sparsity(fitness: np.array, pareto_first_front=None):
    n_objectives = fitness.shape[1]
    if pareto_first_front is not None:
        n_individuals = len(pareto_first_front)
        pareto_fitness = fitness[pareto_first_front]
    else:
        n_individuals = len(fitness)
        pareto_fitness = fitness
    sort_obj = np.sort(pareto_fitness, axis=0).T
    sparsity = 0
    for obj in range(n_objectives):
        for i in range(n_individuals):
            if i > 0:
                sparsity += np.square(sort_obj[obj][i] - sort_obj[obj][i-1])
    if n_individuals > 1:
        sparsity /= (n_individuals - 1)

    return sparsity

def euclidean_distance(solution1, solution2):
    solution1 = np.array(solution1)
    solution2 = np.array(solution2)
    return np.linalg.norm(solution1-solution2)

def calculate_delta(fitness, ref):
    """Calcualte delta metric for biobjective problems

    Args:
        fitness (np.array): fitness of solution1's pareto front
        ref (np.array): fitness of reference's pareto front
    """
    n_solution = len(fitness)
    if n_solution < 2:
        return 1.0
    fitness = sorted(fitness, key=lambda x:x[0])
    ref = sorted(ref, key=lambda x:x[0])
    df = euclidean_distance(fitness[0], ref[0])
    dl = euclidean_distance(fitness[-1], ref[-1])
    di = np.zeros(n_solution-1)
    for i in range(n_solution-1):
        di[i] = euclidean_distance(fitness[i], fitness[i+1])
    d_mean = np.mean(di)
    delta = (df + dl + np.sum(np.abs(di-d_mean))) / (df + dl + (n_solution-1)*d_mean)
    return delta

def compare_delta(fitness1, fitness2):
    """Comparing delta metric on 2 pareto fronts from 2 algorithms

    Args:
        fitness1 (np.array): first pareto front
        fitness2 (np.array): second pareto front

    Returns:
        delta1, delta2: corresponding delta metrics
    """
    all_fitness = np.concatenate((fitness1, fitness2), axis=0)
    ref_front = all_fitness[pareto_front_sort(all_fitness)]
    delta1 = calculate_delta(fitness1, ref_front)
    delta2 = calculate_delta(fitness2, ref_front)
    return delta1, delta2


def create_scalar_list(n_objs, boundary_only=False):
    return np.eye(n_objs)
    if not boundary_only:
        raise ValueError("Non-boundary is unsupported.")