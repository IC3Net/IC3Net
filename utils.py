import numbers
import math
from collections import namedtuple

import numpy as np

import torch
import torch.nn.functional as F
import time
import sys
from torch.autograd import Variable

LogField = namedtuple('LogField', ('data', 'plot', 'x_axis', 'divide_by'))

def merge_stat(src, dest):
    for k, v in src.items():
        if not k in dest:
            dest[k] = v
        elif isinstance(v, numbers.Number):
            dest[k] = dest.get(k, 0) + v
        elif isinstance(v, np.ndarray): # for rewards in case of multi-agent
            dest[k] = dest.get(k, 0) + v
        else:
            if isinstance(dest[k], list) and isinstance(v, list):
                dest[k].extend(v)
            elif isinstance(dest[k], list):
                dest[k].append(v)
            else:
                dest[k] = [dest[k], v]

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def multinomials_log_density(actions, log_probs):
    log_prob = 0
    for i in range(len(log_probs)):
        log_prob += log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
    return log_prob

def multinomials_log_densities(actions, log_probs):
    log_prob = [0] * len(log_probs)
    for i in range(len(log_probs)):
        log_prob[i] += log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
    log_prob = torch.cat(log_prob, dim=-1)
    return log_prob



def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad

class Timer:
    def __init__(self, msg, sync=False):
        self.msg = msg
        self.sync = sync

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print("{}: {} s".format(self.msg, self.interval))

def pca(X, k=2):
    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])


def init_args_for_env(parser):
    env_dict = {
        'levers': 'Levers-v0',
        'number_pairs': 'NumberPairs-v0',
        'predator_prey': 'PredatorPrey-v0',
        'traffic_junction': 'TrafficJunction-v0',
        'starcraft': 'StarCraftWrapper-v0'
    }

    args = sys.argv
    env_name = None
    for index, item in enumerate(args):
        if item == '--env_name':
            env_name = args[index + 1]

    if not env_name or env_name not in env_dict:
        return

    import gym
    import ic3net_envs

    if env_name == 'starcraft':
        import gym_starcraft

    env = gym.make(env_dict[env_name])
    env.init_args(parser)

def display_models(list_models):
    print('='*100)
    print('Model log:\n')
    for model in list_models:
        print(model)
    print('='*100 + '\n')
