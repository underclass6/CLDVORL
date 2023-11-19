import copy
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import gym
import math
import os

def get_gym(env_name, friction, mass):
    """
    Create gym instance and modify mass and friction.
    env_name: name of gym env
    friction: desired friction
    mass: desired torso mass
    returns: gym env
    """
    env = gym.make(env_name)
    env.setFriction(friction)
    env.setMass(mass)
    env.apply_env_modifications() 
    return env
    
def make_results_dir(env_name):
    """
    Create results folder and return path.
    env_name: name of gym env
    returns: results folder path
    """
    if env_name == "Pendulum-v0":
        results_dir = "./results/%s" % (env_name)
        Path(results_dir).mkdir(parents=True, exist_ok=True)
    else:
        date = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
        results_dir = "./results/%s-%s" % (env_name, date)
        Path(results_dir).mkdir(parents=True, exist_ok=True)
    return results_dir


def make_final_results_dir(env_name, model):
    """
    Create results folder and return path.
    env_name: name of gym env
    returns: results folder path
    """

    results_dir = "./results/NEW_final/%s/%s" % (env_name, model)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return results_dir

def make_final_DV_results_dir(env_name, model, target):
    """
    Create results folder and return path.
    env_name: name of gym env
    returns: results folder path
    """
    date = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    # results_dir = "./results/%s/DV_%s/%s/%s/" % (env_name, model, target, date)
    results_dir = "./results/NNN/%s/DV_%s/%s/%s/" % (env_name, model, target, date)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return results_dir

def make_final_DV_results_dir_multiRuns_targ_plot(env_name, model_t, model, target, kl_mode, run_id):
    """
    Create results folder and return path.
    env_name: name of gym env
    returns: results folder path
    """
    results_dir = "./results/IROS_PLOT_SASP/%s/%s/DV%s/%s/%s/%d/" % (env_name, model_t, model, target, kl_mode, run_id)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return results_dir

def make_final_DV_results_dir_multiRuns_targ(env_name, model_t, model, target, kl_mode, run_id):
    """
    Create results folder and return path.
    env_name: name of gym env
    returns: results folder path
    """
    results_dir = "./results/IROS_BEST_SASP/%s/%s/DV%s/%s/%s/%d/" % (env_name, model_t, model, target, kl_mode, run_id)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return results_dir

def make_final_baseline_results_dir_multiRuns(env_name, model, kl_mode, run_id):
    """
    Create results folder and return path.
    env_name: name of gym env
    returns: results folder path
    """
    results_dir = "./results/IROS_NEW/%s/baselines/%s/%s/%d/" % (env_name, model, kl_mode, run_id)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return results_dir

def make_exp_result(env_name, model, target):
    """
    Create results folder and return path.
    env_name: name of gym env
    returns: results folder path
    """
    date = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    results_dir = "./results/exp/NEW_final/%s/DV_%s/%s/%s/" % (env_name, model, target, date)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return results_dir

def make_final_DV_results_paramTuning_dir(env_name, model, param_name, param_value, mtype):
    """
    Create results folder and return path.
    env_name: name of gym env
    returns: results folder path
    """
    # date = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    results_dir = "./results/NNN_final_param/%s/DV_%s/%s/%s/%s/" % (env_name, model, param_name, param_value, mtype)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    return results_dir

def make_dvrl_dir(results_dir, args):
    """
    Create results folder and return path.
    results_dir: base dir
    returns: results folder path
    """
    if args.env == "Pendulum-v0":
        directory = "%s/dvrl_models/Trained_With_Seed_%d_Gamma_%f/" %\
                (results_dir, args.source_seed, args.discount)
    else:
        directory = "%s/dvrl_models/Trained_With_Seed_%d_Friction_%f_Mass_%f_Gamma_%f/" %\
                (results_dir, args.source_seed, args.source_env_friction,
                 args.source_env_mass_torso, args.discount)
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def assert_dir(file_dir):
    """
    Make sure the specified folder exists
    """
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

def get_maximum(input, other):
    """
    Return the larger value of the two tensors
    Note: Both tensors have only one element value
    """
    ans = input if (input > other).item else other
    return ans

def detach(x):
    """
    Detach tensors and return numpy array.
    """
    return x.cpu().detach().numpy()

def concat(states, next_states, actions, rewards, terminals, device):
    """
    returns concatinated state transition tuples
    """
    return torch.FloatTensor(np.hstack((states,
                                             next_states,
                                             actions,
                                             rewards,
                                             terminals))).to(device)

def concat_marginal_information(states, next_states, actions, rewards, terminals,
                                next_states_pred, rewards_pred, terminals_pred, device):
                                # next_states_pred, rewards_pred, actions_pred, terminals_pred, device):
    """
    returns concatinated state transition tuples
    """
    return torch.FloatTensor(np.concatenate((states,
                                             next_states,
                                             actions,
                                             rewards,
                                             terminals,
                                             abs(next_states-next_states_pred),
                                             abs(rewards-rewards_pred),
                                             # abs(actions-actions_pred),
                                             abs(terminals-terminals_pred)), axis=1)).to(device)
    

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def add_noise(states, actions, state_nr, action_nr, batch_size, noise_case='states_actions'):
    num_states  = len(states)
    assert num_states > batch_size
    noisy_samples_indexes = np.random.randint(0, num_states, size=batch_size)
    noisy_states = copy.deepcopy(states)
    noisy_actions = copy.deepcopy(actions)
    mean_states = np.mean(noisy_states, axis = 0)
    mean_actions = np.mean(noisy_actions, axis = 0)
    state_noise_std = mean_states * 0.5
    action_noise_std = mean_states * action_nr
    if noise_case == "states_actions":
        for i in range(mean_states.shape[0]):
            if(state_noise_std[i]>0) and (i in noisy_samples_indexes):
                noisy_states[i] = np.copy(noisy_states[i] + np.random.normal(0, np.absolute(state_noise_std[i]), (noisy_states.shape[1],)))
            if(action_noise_std[i]>0) and (i in noisy_samples_indexes):
                noisy_actions[i] = np.copy(noisy_actions[i] + np.random.normal(0, np.absolute(action_noise_std[i]), (noisy_actions.shape[1],)))
    elif noise_case == "states":
        for i in range(mean_states.shape[0]):
            if(state_noise_std[i]>0) and (i in noisy_samples_indexes):
                noisy_states[i] = np.copy(noisy_states[i] + np.random.normal(0, np.absolute(state_noise_std[i]), (noisy_states.shape[1],)))
    elif noise_case == "actions":
        for i in range(mean_actions.shape[0]):
            if (action_noise_std[i] > 0) and (i in noisy_samples_indexes):
                noisy_actions[i] = np.copy(
                    noisy_actions[i] + np.random.normal(0, np.absolute(action_noise_std[i]), (noisy_actions.shape[1],)))
    else:
        raise ValueError("The given noise_case is not supported!")

    return noisy_states, noisy_actions, noisy_samples_indexes



def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return (TP, FP, TN, FN)



def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)


