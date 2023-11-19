""" Data Valuation based Batch-Constrained Reinforcement Learning """
from itertools import cycle

import scipy.special
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset
from data_valuation.utils.utils import detach, concat, concat_marginal_information
from data_valuation.agents.bcq import BCQ
from data_valuation.agents.bear import BEAR
from data_valuation.agents.TD3_BC import TD3_BC
# from data_valuation.agents.algos import BEAR
from data_valuation.agents.cql_sac import CQLSAC
from data_valuation.utils import utils
import torch.nn.functional as F
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json
import copy
import gym
import os
import time
from pathlib import Path
from collections import deque
import torch.optim.lr_scheduler as lr_scheduler
from numpy.random import default_rng
from numpy.random import Generator, PCG64
from data_valuation.utils.kld import kldiv
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from scipy.special import kl_div


from data_valuation.utils.replay_buffer import ReplayBuffer
from data_valuation.agents.reinforce import REINFORCE

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def discount(r, gamma, normal):
    discount = np.zeros_like(r)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * gamma + r[i]
        discount[i] = G
    # Normalize
    if normal:
        mean = np.mean(discount)
        std = np.std(discount)
        discount = (discount - mean) / (std)
    return discount
    # return discount


def standardize_array(array):
    """
    Z-transformation of the given array
    :param array: (np.ndarray) Data array
    :returns: (np.ndarray) Standarized array
    """
    array = np.array(array)
    std_value = array.std()
    if std_value <= 0:
        std_value = 1e-8
    return (array-array.mean())/std_value

def normalize_array(array):
    """
    Min-Max normalization of the given array
    :param array: (np.ndarray) Data array
    :returns: (np.ndarray) Normalized array
    """
    max_value = np.max(array)
    min_value = np.min(array)
    val = (array-min_value)/(max_value-min_value)
    return val * 2 - 1
    # return val * 0.2 - 0.1


class DVRL(object):
    def __init__(self, source_dataset, target_dataset, device, target_env, results_dir, ex_configs, args={}):
       
        self.dict = vars(args)
        # self.dict = json.dumps(args)
        self.args = args
        assert len(self.dict) > 0
        
        # data
        self.model_dir = utils.make_dvrl_dir(results_dir, args)
        self.state_dim = target_env.observation_space.shape[0]
        self.action_dim = target_env.action_space.shape[0]
        self.action_space = target_env.action_space
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.results_dir = results_dir
        self.device = device
        self.target_env = target_env
        self.env_name = self.dict['env']

        self.dist_epsilon = self.dict['dist_epsilon']
        self.epsilon_minimum = 1.2
        self.epsilon_decay = 0.99

        self.make_target_envs(ex_configs)


        self.g_min = 10000.
        self.g_max = 0.

        self.g_dvbcq_rew_history = []
        self.g_bcq_rew_history = []

        self.expert_actions = None

        if self.dict['rl_model'] == 'BCQ':
            self.expert_model = self.get_bcq()
            self.rl_model = self.get_bcq()
            self.random_model = self.get_bcq()
        elif self.dict['rl_model'] == 'BEAR':
            self.expert_model = self.get_bear()
            self.rl_model = self.get_bear()
            self.random_model = self.get_bear()
        elif self.dict['rl_model'] == 'CQL':
            self.expert_model = self.get_cql()
            self.rl_model = self.get_cql()
            self.random_model = self.get_cql()
        elif self.dict['rl_model'] == 'TD3_BC':
            self.expert_model = self.get_td3bc()
            self.rl_model = self.get_td3bc()
            self.random_model = self.get_td3bc()



        # Used to store samples selected by the value estimator in each outer iteration.
        self.sampling_replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, device)


        self.ev_buf = ReplayBuffer(self.state_dim, self.action_dim, device)

        # Input is a concat of (s, s', a, r, t)
        reinforce_state_dim = self.state_dim * 2 + self.action_dim + 2
        # reinforce_state_dim = self.state_dim * 2 + 2

        reinforce_action_dim = 1 # Outputs probability for each sample
        self.dve_model = REINFORCE(reinforce_state_dim,
                                   reinforce_action_dim,
                                   layers=self.dict['reinforce_layers'],
                                   args=self.dict).to(device)
        # optimizers
        self.dve_optimizer = optim.Adam(self.dve_model.parameters(), self.dict['dve_lr'])

        # self.scheduler = lr_scheduler.ExponentialLR(self.dve_optimizer, gamma=0.999)

        # if self.dict["env"] == "HalfCheetah-v3":
        #     self.source_dataset =  self.get_final_transitions(source_dataset)
        #     self.target_dataset =  self.get_final_transitions(target_dataset)

        # self.source_dataset.shuffle()
        # self.target_dataset.shuffle()

    def make_target_envs(self, ex_configs):
        self.target1_env = utils.get_gym(self.dict["env"],
                                         ex_configs[self.dict["env"]]["target_1"][0],
                                         ex_configs[self.dict["env"]]["target_1"][1])
        self.target2_env = utils.get_gym(self.dict["env"],
                                         ex_configs[self.dict["env"]]["target_2"][0],
                                         ex_configs[self.dict["env"]]["target_2"][1])
        self.target3_env = utils.get_gym(self.dict["env"],
                                         ex_configs[self.dict["env"]]["target_3"][0],
                                         ex_configs[self.dict["env"]]["target_3"][1])
        self.target4_env = utils.get_gym(self.dict["env"],
                                         ex_configs[self.dict["env"]]["target_4"][0],
                                         ex_configs[self.dict["env"]]["target_4"][1])



    def get_final_transitions(self, dataset):
        rbuf = ReplayBuffer(self.target_env.observation_space.shape[0], self.target_env.action_space.shape[0], self.device, max_size=dataset.size)

        states, actions, next_states, rewards, terminals = dataset.sample(batch_size=len(dataset), to_device=False)
        i = 0
        for s, a, s_, r, t in zip(states, actions, next_states, rewards, terminals):
            if (i == 1000):
                done = 1-t
                rbuf.add(s, a, s_, r, done)
                i = 0
            i+=1
        return rbuf


    def get_final_buffer(self, dataset, sel_prob, sel_vec, threshold=0.1, d_t="prob", exclude_low=True):
        rbuf = ReplayBuffer(self.target_env.observation_space.shape[0], self.target_env.action_space.shape[0], self.device, max_size=dataset.size)

        states, actions, next_states, rewards, terminals = dataset.sample(batch_size=len(dataset), to_device=False)
        if d_t == "prob":
            data = sel_prob
        elif d_t == "sel":
            data = sel_vec
        for s, a, r, s_, t, sp in zip(states, actions, rewards, next_states, terminals, data):
            if d_t=="prob":
                if exclude_low:
                    if sp >= threshold:
                        rbuf.add(s, a, s_, r, 1-t)
                else:
                    if sp <= (1.0 - threshold):
                        rbuf.add(s, a, s_, r, 1-t)
            elif d_t=="sel":
                if int(sp) == 1:
                    rbuf.add(s, a, s_, r, 1-t)
        return rbuf

    def get_bcq(self):
        """
        returns new bcq instance
        """
        return BCQ(self.state_dim, 
                   self.action_dim, 
                   self.action_space,
                   self.device, 
                   self.dict['discount'], 
                   self.dict['tau'], 
                   self.dict['lmbda'], 
                   self.dict['phi'])
    def get_bear(self):
        return BEAR(2, self.state_dim,
               self.action_dim,
               self.action_space.high,
               delta_conf=0.1,
               use_bootstrap=False,
               version=self.dict['version'],
               lambda_=float(self.dict['bear_lambda']),
               threshold=float(self.dict['bear_threshold']),
               mode=self.dict['mode'],
               num_samples_match=self.dict['num_samples_match'],
               mmd_sigma=self.dict['mmd_sigma'],
               lagrange_thresh=self.dict['lagrange_thresh'],
               use_kl=(True if self.dict['distance_type'] == "KL" else False),
               use_ensemble=(False if self.dict['use_ensemble_variance'] == "False" else True),
               kernel_type=self.dict['kernel_type'],
               device=self.device)

    def get_cql(self):
        return CQLSAC(self.state_dim,
                        self.action_dim,
                        self.dict['tau'],
                        self.dict['cql_hidden_size'],
                        self.dict['learning_rate'],
                        self.dict['temperature'],
                        self.dict['with_lagrange'],
                        self.dict['cql_weight'],
                        self.dict['target_action_gap'],
                        self.device)

    def get_td3bc(self):
        return TD3_BC(self.state_dim,
                      self.action_dim,
                      self.action_space.high,
                      self.dict['discount'],
                      self.dict['tau'],
                      self.dict['policy_noise'],
                      self.dict['noise_clip'],
                      self.dict['policy_freq'],
                      self.dict['alpha'],
                      self.device)

    def to_device(self, x):
        """
        Copy x to GPU
        """
        return torch.FloatTensor(x).to(self.device)


    def load_train_baselines(self):
        # if self.dict['rl_model'] == 'BCQ':
        if self.dict['expert_path'] != None:
            print('loading train-baseline model.')
            # if self.dict['target_env_choice'] == 'target_1':
            self.expert_model.load(self.dict['expert_path'], type="best_t1")
            # elif self.dict['target_env_choice'] == 'target_2':
            #     self.expert_model.load(self.dict['expert_path'], type="best_t2")
            # elif self.dict['target_env_choice'] == 'target_3':
            #     self.expert_model.load(self.dict['expert_path'], type="best_t3")
            # elif self.dict['target_env_choice'] == 'target_4':
            #     self.expert_model.load(self.dict['expert_path'], type="best_t4")
        else:
            print('start training baseline model.')
            self.train_dvbca(self.source_dataset, m_type="expert")
        # elif self.dict['rl_model'] == 'CQL':
        #     if self.dict['expert_path'] != None:
        #         print('loading train-baseline model.')
        #         self.expert_model.load(self.dict['expert_path'], type="final")
        #     else:
        #         print('start training baseline model.')
        #         self.train_dvbca(self.source_dataset, m_type="expert")
        # elif self.dict['rl_model'] == 'BEAR':
        #     if self.dict['expert_path'] != None:
        #         print('loading train-baseline model.')
        #         self.expert_model.load(self.dict['expert_path'], type="final")
        #     else:
        #         print('start training baseline model.')
        #         self.train_dvbca(self.source_dataset, m_type="expert")


    def load_train_dvbcq(self):
        # if self.dict['rl_model'] == 'BCQ':
        # if self.dict['rl_model_t'] == 'fresh':
        #     self.rl_model = self.get_bcq()
        if self.dict['rl_model_t'] == 'expert' and self.dict['expert_path'] != None:
            # self.rl_model = self.get_bcq()
            # if self.dict['target_env_choice'] == 'target_1':
            self.rl_model.load(self.dict['expert_path'], type="best_t1")
            # elif self.dict['target_env_choice'] == 'target_2':
            #     self.rl_model.load(self.dict['expert_path'], type="best_t2")
            # elif self.dict['target_env_choice'] == 'target_3':
            #     self.rl_model.load(self.dict['expert_path'], type="best_t3")
            # elif self.dict['target_env_choice'] == 'target_4':
            #     self.rl_model.load(self.dict['expert_path'], type="best_t4")
            # else:
            #     self.rl_model = self.get_bcq()

        # elif self.dict['rl_model'] == "BEAR":
        #     if self.dict['rl_model_t'] == 'fresh':
        #         self.rl_model = self.get_bear()
        #     elif self.dict['rl_model_t'] == 'expert' and self.dict['expert_path'] != None:
        #         self.rl_model = self.get_bear()
        #         if self.dict['target_env_choice'] == 'target_1':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t1")
        #         elif self.dict['target_env_choice'] == 'target_2':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t2")
        #         elif self.dict['target_env_choice'] == 'target_3':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t3")
        #         elif self.dict['target_env_choice'] == 'target_4':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t4")
        #
        # elif self.dict['rl_model'] == "CQL":
        #     if self.dict['rl_model_t'] == 'fresh':
        #         self.rl_model = self.get_cql()
        #     elif self.dict['rl_model_t'] == 'expert' and self.dict['expert_path'] != None:
        #         self.rl_model = self.get_cql()
        #         if self.dict['target_env_choice'] == 'target_1':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t1")
        #         elif self.dict['target_env_choice'] == 'target_2':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t2")
        #         elif self.dict['target_env_choice'] == 'target_3':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t3")
        #         elif self.dict['target_env_choice'] == 'target_4':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t4")
        #
        #     # elif self.dict['rl_model_t'] == 'expert':
        #     #     self.rl_model = copy.deepcopy(self.expert_model)
        # elif self.dict['rl_model'] == "TD3_BC":
        #     if self.dict['rl_model_t'] == 'fresh':
        #         self.rl_model = self.get_cql()
        #     elif self.dict['rl_model_t'] == 'expert' and self.dict['expert_path'] != None:
        #         self.rl_model = self.get_cql()
        #         if self.dict['target_env_choice'] == 'target_1':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t1")
        #         elif self.dict['target_env_choice'] == 'target_2':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t2")
        #         elif self.dict['target_env_choice'] == 'target_3':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t3")
        #         elif self.dict['target_env_choice'] == 'target_4':
        #             self.rl_model.load(self.dict['expert_path'], type="best_t4")

    def train_baseline(self, dataset, m_type="BCQ", eval_episodes=100):
        target1_avg_reward = np.array([])
        target1_avg_reward_max = np.array([])
        target2_avg_reward = np.array([])
        target2_avg_reward_max = np.array([])
        target3_avg_reward = np.array([])
        target3_avg_reward_max = np.array([])
        target4_avg_reward = np.array([])
        target4_avg_reward_max = np.array([])

        max_reward_t1 = -10000.0
        c_max_reward_t1 = -10000.0
        max_reward_t2 = -10000.0
        c_max_reward_t2 = -10000.0
        max_reward_t3 = -10000.0
        c_max_reward_t3 = -10000.0
        max_reward_t4 = -10000.0
        c_max_reward_t4 = -10000.0

        training_iters = 0
        model_path = f"{self.model_dir}/{m_type}/"
        Path(model_path).mkdir(parents=True, exist_ok=True)

        if m_type == "BCQ":
            model = self.get_bcq()
            best_model_t1 = self.get_bcq()
            best_model_t2 = self.get_bcq()
            best_model_t3 = self.get_bcq()
            best_model_t4 = self.get_bcq()
        elif m_type == "BEAR":
            model = self.get_bear()
            best_model_t1 = self.get_bear()
            best_model_t2 = self.get_bear()
            best_model_t3 = self.get_bear()
            best_model_t4 = self.get_bear()
        elif m_type == "TD3_BC":
            model = self.get_td3bc()
            best_model_t1 = self.get_td3bc()
            best_model_t2 = self.get_td3bc()
            best_model_t3 = self.get_td3bc()
            best_model_t4 = self.get_td3bc()
        elif m_type == "CQL":
            model = self.get_cql()
            best_model_t1 = self.get_cql()
            best_model_t2 = self.get_cql()
            best_model_t3 = self.get_cql()
            best_model_t4 = self.get_cql()

        for i in tqdm(range(0, int(self.dict['training_max_timesteps']), int(self.dict['bl_eval_freq']))):
            if self.dict['rl_model'] == 'BCQ':
                model.train(replay_buffer=dataset,
                            iterations=int(self.dict['bl_eval_freq']),
                            batch_size=self.dict['bcq_batch_size'],
                            random_s=True,
                            disable_tqdm=False)
            elif self.dict['rl_model'] == 'BEAR':
                model.train(replay_buffer=dataset,
                            iterations=int(self.dict['bl_eval_freq']),
                            batch_size=self.dict['bear_batch_size'],
                            random_s=True,
                            disable_tqdm=False)
            elif self.dict['rl_model'] == 'TD3_BC':
                model.train(replay_buffer=dataset,
                            iterations=int(self.dict['bl_eval_freq']),
                            batch_size=self.dict['td3bc_batch_size'],
                            random_s=True,
                            disable_tqdm=False)
            elif self.dict['rl_model'] == "CQL":
                model.train(replay_buffer=dataset,
                            iterations=int(self.dict['bl_eval_freq']),
                            batch_size=self.dict['cql_batch_size'],
                            random_s=True,
                            disable_tqdm=False)

            r_target1 = self.eval_policy(model, eval_episodes=eval_episodes, t_env="target_1")
            r_target2 = self.eval_policy(model, eval_episodes=eval_episodes, t_env="target_2")
            r_target3 = self.eval_policy(model, eval_episodes=eval_episodes, t_env="target_3")
            r_target4 = self.eval_policy(model, eval_episodes=eval_episodes, t_env="target_4")


            if r_target1 > max_reward_t1:
                max_reward_t1 = r_target1
                c_max_reward_t1 = r_target1
                model.save(model_path, type="best_t1")
                print(f"New best {m_type}-baseline model with reward {r_target1} on {self.dict['target_env_choice']} has been saved in: {model_path}")
            else:
                if self.dict['rl_model'] == 'BCQ':
                    best_model_t1.load(model_path, type="best_t1")
                elif self.dict['rl_model'] == 'BEAR':
                    best_model_t1.load(model_path, type="best_t1")
                elif self.dict['rl_model'] == 'CQL':
                    best_model_t1.load(model_path, type="best_t1")
                elif self.dict['rl_model'] == 'TD3_BC':
                    best_model_t1.load(model_path, type="best_t1")
                c_max_reward_t1 = self.eval_policy(best_model_t1, eval_episodes=eval_episodes, t_env="target_1")

            if r_target2 > max_reward_t2:
                max_reward_t2 = r_target2
                c_max_reward_t2 = r_target2
                model.save(model_path, type="best_t2")
                print(f"New best {m_type}-baseline model with reward {r_target2} on {self.dict['target_env_choice']} has been saved in: {model_path}")
            else:
                if self.dict['rl_model'] == 'BCQ':
                    best_model_t2.load(model_path, type="best_t2")
                elif self.dict['rl_model'] == 'BEAR':
                    best_model_t2.load(model_path, type="best_t2")
                elif self.dict['rl_model'] == 'CQL':
                    best_model_t2.load(model_path, type="best_t2")
                elif self.dict['rl_model'] == 'TD3_BC':
                    best_model_t2.load(model_path, type="best_t2")
                c_max_reward_t2 = self.eval_policy(best_model_t2, eval_episodes=eval_episodes, t_env="target_2")

            if r_target3 > max_reward_t3:
                max_reward_t3 = r_target3
                c_max_reward_t3 = r_target3
                model.save(model_path, type="best_t3")
                print(f"New best {m_type}-baseline model with reward {r_target3} on {self.dict['target_env_choice']} has been saved in: {model_path}")
            else:
                if self.dict['rl_model'] == 'BCQ':
                    best_model_t3.load(model_path, type="best_t3")
                elif self.dict['rl_model'] == 'BEAR':
                    best_model_t3.load(model_path, type="best_t3")
                elif self.dict['rl_model'] == 'CQL':
                    best_model_t3.load(model_path, type="best_t3")
                elif self.dict['rl_model'] == 'TD3_BC':
                    best_model_t3.load(model_path, type="best_t3")
                c_max_reward_t3 = self.eval_policy(best_model_t3, eval_episodes=eval_episodes, t_env="target_3")

            if r_target4 > max_reward_t4:
                max_reward_t4 = r_target4
                c_max_reward_t4 = r_target4
                model.save(model_path, type="best_t4")
                print(f"New best {m_type}-baseline model with reward {r_target4} on {self.dict['target_env_choice']} has been saved in: {model_path}")
            else:
                if self.dict['rl_model'] == 'BCQ':
                    best_model_t4.load(model_path, type="best_t4")
                elif self.dict['rl_model'] == 'BEAR':
                    best_model_t4.load(model_path, type="best_t4")
                elif self.dict['rl_model'] == 'CQL':
                    best_model_t4.load(model_path, type="best_t4")
                elif self.dict['rl_model'] == 'TD3_BC':
                    best_model_t4.load(model_path, type="best_t4")
                c_max_reward_t4 = self.eval_policy(best_model_t4, eval_episodes=eval_episodes, t_env="target_4")

            target1_avg_reward = np.append(target1_avg_reward, r_target1)
            target1_avg_reward_max = np.append(target1_avg_reward_max, c_max_reward_t1)
            target2_avg_reward = np.append(target2_avg_reward, r_target2)
            target2_avg_reward_max = np.append(target2_avg_reward_max, c_max_reward_t2)
            target3_avg_reward = np.append(target3_avg_reward, r_target3)
            target3_avg_reward_max = np.append(target3_avg_reward_max, c_max_reward_t3)
            target4_avg_reward = np.append(target4_avg_reward, r_target4)
            target4_avg_reward_max = np.append(target4_avg_reward_max, c_max_reward_t4)


            print(f"Training iterations: {training_iters} | Env: target1 | Avg Reward: {r_target1}")
            print(f"Training iterations: {training_iters} | Env: target2 | Avg Reward: {r_target2}")
            print(f"Training iterations: {training_iters} | Env: target3 | Avg Reward: {r_target3}")
            print(f"Training iterations: {training_iters} | Env: target4 | Avg Reward: {r_target4}")


            np.save(f'{self.results_dir}/{m_type}_avg_reward_target_1', target1_avg_reward, allow_pickle=True)
            np.save(f'{self.results_dir}/{m_type}_avg_reward_max_target_1', target1_avg_reward_max, allow_pickle=True)
            np.save(f'{self.results_dir}/{m_type}_avg_reward_target_2', target2_avg_reward, allow_pickle=True)
            np.save(f'{self.results_dir}/{m_type}_avg_reward_max_target_2', target2_avg_reward_max, allow_pickle=True)
            np.save(f'{self.results_dir}/{m_type}_avg_reward_target_3', target3_avg_reward, allow_pickle=True)
            np.save(f'{self.results_dir}/{m_type}_avg_reward_max_target_3', target3_avg_reward_max, allow_pickle=True)
            np.save(f'{self.results_dir}/{m_type}_avg_reward_target_4', target4_avg_reward, allow_pickle=True)
            np.save(f'{self.results_dir}/{m_type}_avg_reward_max_target_4', target4_avg_reward_max, allow_pickle=True)

            training_iters += int(self.dict['bl_eval_freq'])

        model.save(model_path, type="final")
        print(f"Final {m_type}-baseline model has been saved in: {model_path}")


    def train_dvbca(self, dataset, m_type="expert", eval_episodes=100):

        bl_avg_reward = np.array([])
        bl_avg_reward_max = np.array([])
        max_reward = -1000.0
        c_max_reward = -1000.0

        it_idx = 0

        training_iters = 0
        model_path = f"{self.model_dir}/DVBCQ/"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        values = []
        keys = []
        # if data == "train":

        best_model_e = False


        if m_type == "BCQ":
            model = self.get_bcq()
            best_model = self.get_bcq()
        elif m_type == "BEAR":
            model = self.get_bear()
            best_model = self.get_bear()
        elif m_type == "TD3_BC":
            model = self.get_td3bc()
            best_model = self.get_td3bc()
        elif m_type == "CQL":
            model = self.get_cql()
            best_model = self.get_cql()


        states, _, _, rewards, _ = dataset.sample(dataset.size)

        for r in rewards:
            keys.append(str(detach(r)[0]))


        # while training_iters < self.dict['bl_training_max_timesteps']:
        for i in tqdm(range(0, int(self.dict['training_max_timesteps']), int(self.dict['bl_eval_freq']))):
            if self.dict['rl_model'] == 'BCQ':
                model.train(replay_buffer=dataset,
                            iterations=int(self.dict['bl_eval_freq']),
                            batch_size=self.dict['bcq_batch_size'],
                            random_s=True,
                        disable_tqdm=False)
            elif self.dict['rl_model'] == 'BEAR':
                model.train(replay_buffer=dataset,
                            iterations=int(self.dict['eval_freq']),
                            batch_size=self.dict['bear_batch_size'],
                            random_s=True,
                            disable_tqdm=False)
            elif self.dict['rl_model'] == 'TD3_BC':
                model.train(replay_buffer=dataset,
                            iterations=int(self.dict['bl_eval_freq']),
                            batch_size=self.dict['td3bc_batch_size'],
                            random_s=True,
                            disable_tqdm=False)
            elif self.dict['rl_model'] == "CQL":
                model.train(replay_buffer=dataset,
                            iterations=int(self.dict['bl_eval_freq']),
                            # batch_size=self.dict['batch_size'],
                            # batch_size=self.dict['mini_batch_size'],
                            batch_size=self.dict['cql_batch_size'],
                            random_s=True,
                            disable_tqdm=False)

            r = self.eval_policy(model, eval_episodes=eval_episodes, t_env=self.dict["target_env_choice"])


            if r > max_reward:
                max_reward = r
                c_max_reward = r
                model.save(model_path, type="best")
                print(f"New best DVBCQ model with reward {r}  has been saved in: {model_path}")
            else:
                if self.dict['rl_model'] == 'BCQ':
                    best_model.load(model_path, type="best")
                elif self.dict['rl_model'] == 'BEAR':
                    best_model.load(model_path, type="best")
                elif self.dict['rl_model'] == 'CQL':
                    best_model.load(model_path, type="best")
                c_max_reward = self.eval_policy(best_model, eval_episodes=eval_episodes, t_env=self.dict["target_env_choice"])



            bl_avg_reward = np.append(bl_avg_reward, r)
            bl_avg_reward_max = np.append(bl_avg_reward_max, c_max_reward)

            training_iters += int(self.dict['bl_eval_freq'])

            print(f"Env: {self.dict['env']} | Training iterations: {training_iters} | Avg Reward: {r}")


            np.save(f'{self.results_dir}/DV{self.dict["rl_model"]}_avg_reward_{self.dict["target_env_choice"]}', bl_avg_reward, allow_pickle=True)
            np.save(f'{self.results_dir}/DV{self.dict["rl_model"]}_avg_reward_max_{self.dict["target_env_choice"]}', bl_avg_reward_max, allow_pickle=True)

        model.save(model_path, type="final")
        print(f"Final {m_type}-baseline model has been saved in: {model_path}")

        return np.max(bl_avg_reward_max)

    def train_dve(self, dve_model, dve_optimizer, x, s_input, reward):
        """
         Training data value estimator
         s_input: selection probability
         x_input: Sample tuple
         reward_input: reward signal for training the reinforce agent
        """
        est_data_value = dve_model(x)
        dve_optimizer.zero_grad()


        one = torch.ones_like(est_data_value, dtype=est_data_value.dtype)
        prob = torch.sum(s_input * torch.log(est_data_value + self.dict['epsilon']) + \
                         (one - s_input) * \
                         torch.log(one - est_data_value + self.dict['epsilon']))

        zero = torch.Tensor([0.0])
        zero = zero.to(est_data_value.device)

        # 1e3= multiplier for the regularizer
        loss = (-reward * prob) + \
                   1e3 * torch.maximum(torch.mean(est_data_value) - self.dict['threshold'], zero) + \
                   1e3 * torch.maximum(1 - self.dict['threshold'] - torch.mean(est_data_value), zero)

        loss.backward()
        dve_optimizer.step()


    def test_BCQ(self, m_type, eval_episodes=100, env="target"):
        # training_iters = 0
        # test_avg_reward = np.array([])

        # for i in tqdm(range(0, int(self.dict['max_timesteps']), int(self.dict['bl_eval_freq']))):
        # if self.dict['rl_model'] == 'BCQ':
            # actor_loss, critic_loss, vae_loss = model.train(
        # baseline_model = self.get_bcq()
        if m_type == "expert":
            model = self.expert_model
            model.load(self.dict['expert_path'], type="best")
        elif m_type == "BCQ":
            model = self.get_bcq()
            best_model = self.get_bcq()
        elif m_type == "BEAR":
            model = self.get_bear()
            best_model = self.get_bear()
        elif m_type == "CQL":
            model = self.get_cql()
            best_model = self.get_cql()
        elif m_type == "TD3_BC":
            model = self.get_td3bc()
            best_model = self.get_td3bc()
        else:
            model = self.get_bcq()
            best_model = self.get_bcq()

        r_dvbcq = self.eval_policy(model, eval_episodes=eval_episodes, env=env)
        print("Avg reward of DVBCQ on target buffer: ", r_dvbcq)

    def adaptiveDistanceEpsilon(self):
        """
        Adaptive Distance Epsilon
        at every step we decrease the epsilon
        """
        if self.dist_epsilon > self.epsilon_minimum:
            self.dist_epsilon *= self.epsilon_decay
    def train(self):
        """
        RL model and DVE iterative training
        """
        evaluations_df = []
        # List for storing model evaluaions after each outer iteration
        evaluations = np.array([])
        bcq_evals = np.array([])
        bcq_rews = np.array([])
        dvbcq_evals = np.array([])
        dvbcq_rews = np.array([])
        baseline_losses = np.array([])
        mean_losses = np.array([])
        reinforce_rewards = np.array([])

        probs = np.array([])
        est_dv_curr_values = np.array([])

        if self.dict['rl_model'] == "BCQ":
            best_model = self.get_bcq()
        elif self.dict['rl_model'] == "BEAR":
            best_model = self.get_bear()
        elif self.dict['rl_model'] == "CQL":
            best_model = self.get_cql()
        elif self.dict['rl_model'] == "TD3_BC":
            best_model = self.get_td3bc()


        final_DVBCQ_pef = np.array([])
        final_DVBCQ_pef_max = np.array([])
        best_perf = -1000.0
        mc_perf = -1000.0

        rew_deque = deque(maxlen=20)

        update_flag = False
        best_reward_sig = 0.

        reward_signal_history = np.array([0.0])
        baseline_history = np.array([0.0])
        rl_model_rew_history = np.array([])
        rl_model_actual_rew_history = np.array([])
        # rng = default_rng()
        rng = Generator(PCG64(12345))


        dvbcq_avg_rew = 0.0
        dvbcq_actual_rew = 0.0
        bcq_avg_rew = 0.0
        bcq_actual_rew = 0.0
        reward_history = np.array([])

        self.expert_perf = 0.0

        best_dvbcq_perf = 0.0


        # self.load_train_baselines()
        # self.load_train_dvbcq()

        # bcq_avg_rew, bcq_actual_rew = self.normalized_eval_policy(self.expert_model, model_name="CQL", eval_episodes=20, seed_num=self.dict['target_seed'])


        # bcq_avg_rew = self.eval_policy_byAction(self.expert_model, self.target_dataset, model_name="BCQ", batch_size=self.dict["eval_batch_size"], eps=self.dist_epsilon)



        s_states, s_actions, s_next_states, s_rewards, s_terminals = self.source_dataset.sample(batch_size=self.source_dataset.size, to_device=False)
        t_states, t_actions, t_next_states, t_rewards, t_terminals = self.target_dataset.sample(batch_size=10000, to_device=False)





        # if self.dict['kl_mode'] == "state":
        #     bcq_avg_rew = kldiv(s_states, t_states)
        # elif self.dict['kl_mode'] == "action":
        #     bcq_avg_rew = kldiv(s_actions, t_actions)
        # elif self.dict['kl_mode'] == "state_action":
        #     bcq_avg_rew = kldiv(np.concatenate((s_states, s_actions), axis=1), np.concatenate((t_states, t_actions), axis=1))
        # elif self.dict['kl_mode'] == "state_action_nextstate":
        #     bcq_avg_rew = kldiv(np.concatenate((s_states, s_actions, s_next_states), axis=1), np.concatenate((t_states, t_actions, t_next_states), axis=1))
        # elif self.dict['kl_mode'] == "custom1":
        #     bcq_avg_rew = kldiv(np.concatenate((s_states, s_actions, s_next_states), axis=1), np.concatenate((t_states, t_actions, t_next_states), axis=1))
        # elif self.dict['kl_mode'] == "all":
        #     bcq_avg_rew = kldiv(np.concatenate((s_states, s_actions, s_rewards, s_next_states, s_terminals), axis=1), np.concatenate((t_states, t_actions, t_rewards, t_next_states, t_terminals), axis=1))

        # self.dict['kl_mode'] == "custom1":
        # bcq_s = self.eval_policy_KLD(self.target_dataset, s_states, model_name="DVBCQ",
        #                                batch_size=self.dict["eval_batch_size"], mode="state")

        # bcq_sasp = self.eval_policy_KLD(self.target_dataset, np.concatenate((t_states, t_actions, t_next_states), axis=1), mode="state_action_nextstate")

        # bcq_avg_rew = bcq_sasp - bcq_s
        # bcq_avg_rew = 1.0/bcq_avg_rew
        bcq_avg_rew = 0.0

        # org_bcq_avg_rew = self.eval_policy(self.expert_model, eval_episodes=10, seed_num=100, t_env=self.dict["target_env_choice"])
        # reward_history = np.append(reward_history, dvbcq_avg_rew)

        self.expert_perf = bcq_actual_rew
        update_dve = True

        m_avg = MovingAvg(20)
        m_avg_v = 0.0

        # start training
        # for iteration in tqdm(range(0, self.train_dataset.size, self.dict["batch_size"])):
        # for iteration in tqdm(range(0, self.dict['outer_iterations'], self.dict["eval_freq"])):

        # if self.dict['trained_dve_path'] != None:
        #     self.load_dve(self.dict['trained_dve_path'], type="final")
        train_dve = False

        for iteration in tqdm(range(self.dict['outer_iterations'])):
            # if iteration == 5000:
            #     train_dve = True
        # for iteration in tqdm(range(self.source_dataset.trajs_size)):

            t_start = time.time()

            states, actions, next_states, rewards, terminals = self.source_dataset.sample(self.dict['batch_size'], to_device=False) #O
            # states, actions, next_states, rewards, terminals = self.source_dataset.sample_n_trajectories(traj_num=10, to_device=False) #O
            # states, actions, next_states, rewards, terminals = self.source_dataset.sample_trajectory(ind=iteration, to_device=False) #O


            dvrl_input = concat(states, next_states, actions, rewards, terminals, self.device)
            # dvrl_input = torch.FloatTensor(np.hstack((states, next_states, rewards, terminals))).to(self.device)
            # dvrl_input = torch.FloatTensor(states).to(self.device)

            est_dv_curr = self.dve_model(dvrl_input)


            # Samples the selection probability
            # sel_prob_curr = rng.binomial(1, detach(est_dv_curr), est_dv_curr.shape)
            #
            # # Exception (When selection probability is 0)
            # if np.sum(sel_prob_curr) == 0:
            #     est_dv_curr = 0.5 * np.ones(np.shape(detach(est_dv_curr)))
            #     # sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)
            #     sel_prob_curr = rng.binomial(1, est_dv_curr, est_dv_curr.shape)
            #     est_dv_curr = self.to_device(est_dv_curr)
            #
            #
            # est_dv_curr_values = np.append(est_dv_curr_values, detach(torch.as_tensor(est_dv_curr)).flatten())
            # probs = np.append(probs, sel_prob_curr.flatten())

            # print(detach(torch.as_tensor(est_dv_curr)).flatten())
            # print(sel_prob_curr.flatten())


            # Reset (empty) the sampling buffer, and then add a samples based on the selection probabiliti
            # es.
            # self.reset_sampling_buffer(states, actions, rewards, next_states, terminals, sel_prob_curr.flatten())
            # self.reset_sampling_buffer(states, actions, rewards, next_states, terminals, detach(torch.as_tensor(est_dv_curr)).flatten())

            # print(f"Size of sampling buffer: {self.sampling_replay_buffer.size} (out of {len( sel_prob_curr.flatten())} selected)")

            # if iteration > 100:
            #     self.rl_model.load(self.model_dir, type="best")

            # train rl_model with sampled batch data for inner_iterations
            # print("Training Selective Model")
            # if self.dict['rl_model'] == 'BCQ':
            #
            #     # bcq_avg_rew, bcq_actual_rew = self.normalized_eval_policy(self.bl_train_model, "BCQ", eval_episodes=3, seed_num=seed_num)
            #     # bcq_avg_rew, bcq_actual_rew = self.eval_policy_byAction(self.bl_train_model, self.target_dataset, "BCQ")
            #
            #     self.rl_model.train(replay_buffer=self.sampling_replay_buffer,
            #                             # iterations=self.dict['inner_iterations'],
            #                             # iterations=int(self.dict['batch_size']/self.dict['mini_batch_size']),
            #                             iterations=int(self.sampling_replay_buffer.size/self.dict['mini_batch_size'])+1,
            #                             # iterations=1,
            #                             # iterations=int(self.dict['eval_freq']),
            #                             # batch_size=max(self.dict['mini_batch_size'],self.sampling_replay_buffer.size))
            #                             batch_size=min(self.dict['mini_batch_size'],self.sampling_replay_buffer.size))
            #
            # elif self.dict['rl_model'] == "BEAR":
            #     self.rl_model.train(replay_buffer=self.sampling_replay_buffer,
            #                 iterations=self.dict['inner_iterations'],
            #                 batch_size=min(self.dict['mini_batch_size'],self.sampling_replay_buffer.size),
            #                 disable_tqdm=True)
            #
            # elif self.dict['rl_model'] == "CQL":
            #
            #     self.rl_model.train(replay_buffer=self.sampling_replay_buffer,
            #                 # iterations=int(self.sampling_replay_buffer.size/self.dict['mini_batch_size'])+1,
            #                 # iterations=self.dict['inner_iterations'],
            #                 iterations=int(self.sampling_replay_buffer.size / self.dict['mini_batch_size']) + 1,
            #                 # batch_size=min(self.dict['mini_batch_size'],self.sampling_replay_buffer.size),
            #                 batch_size=min(self.dict['mini_batch_size'],self.sampling_replay_buffer.size),
            #                 disable_tqdm=True)
            # elif self.dict['rl_model'] == "TD3_BC":
            #
            #     self.rl_model.train(replay_buffer=self.sampling_replay_buffer,
            #                 # iterations=int(self.sampling_replay_buffer.size/self.dict['mini_batch_size'])+1,
            #                 # iterations=self.dict['inner_iterations'],
            #                 iterations=int(self.sampling_replay_buffer.size / self.dict['mini_batch_size']) + 1,
            #                 # batch_size=min(self.dict['mini_batch_size'],self.sampling_replay_buffer.size),
            #                 batch_size=min(self.dict['mini_batch_size'],self.sampling_replay_buffer.size),
            #                 disable_tqdm=True)
            #
            # # dvbcq_avg_rew, dvbcq_actual_rew = self.normalized_eval_policy(self.rl_model, "DVCQL", eval_episodes=3, seed_num=self.dict['target_seed'])
            #
            # dvbcq_avg_rew = self.eval_policy_byAction(self.rl_model, self.target_dataset, model_name="DVBCQ", batch_size=self.dict["eval_batch_size"], eps=self.dist_epsilon)


            if self.dict['kl_mode'] == "state":
                dvbcq_avg_rew = self.eval_policy_KLD(self.target_dataset, states, model_name="DVBCQ", batch_size=self.dict["eval_batch_size"], mode="state")
            elif self.dict['kl_mode'] == "action":
                dvbcq_avg_rew = self.eval_policy_KLD(self.target_dataset, actions, model_name="DVBCQ", batch_size=self.dict["eval_batch_size"], mode="action")
            elif self.dict['kl_mode'] == "state_action":
                dvbcq_avg_rew = self.eval_policy_KLD(self.target_dataset, np.concatenate((states, actions), axis=1),model_name="DVBCQ", batch_size=self.dict["eval_batch_size"], mode="state_action")
            elif self.dict['kl_mode'] == "state_action_nextstate":
                dvbcq_avg_rew = self.eval_policy_KLD(self.target_dataset, np.concatenate((states, actions, next_states), axis=1),
                                                     model_name="DVBCQ", batch_size=self.dict["eval_batch_size"], mode="state_action_nextstate")
            elif self.dict['kl_mode'] == "custom1":
                dvbcq_s = self.eval_policy_KLD(self.target_dataset, states, model_name="DVBCQ",
                                                     batch_size=self.dict["eval_batch_size"], mode="state")

                dvbcq_sasp = self.eval_policy_KLD(self.target_dataset,
                                                     np.concatenate((states, actions, next_states), axis=1),
                                                     model_name="DVBCQ", batch_size=self.dict["eval_batch_size"], mode="state_action_nextstate")

                dvbcq_avg_rew = dvbcq_sasp - dvbcq_s
            elif self.dict['kl_mode'] == "all":
                dvbcq_avg_rew = self.eval_policy_KLD(self.target_dataset, np.concatenate((states, actions, rewards, next_states, terminals), axis=1),model_name="DVBCQ", batch_size=self.dict["eval_batch_size"])
            #



            # if dvbcq_avg_rew > best_dvbcq_perf:
            #     best_dvbcq_perf = dvbcq_avg_rew
            #
            #     self.rl_model.save(self.model_dir, type="best")
            #     self.save_dve(self.model_dir, type="best")
            #     print(f"New best DVBCQ model with reward {dvbcq_avg_rew}  has been saved in: {self.model_dir}")




            if m_avg.size >= 1:
                mean = m_avg.average()

                # std = m_avg.std()
                m_avg_v = mean
                # m_avg_v /= std
            else:
                m_avg_v = 0.0


            # Compute reward for the Reinforce agent.
            # The lower the mean_loss, the larger the reward.
            # reinforce_reward = m_avg_v


            # if self.dict["moving_average_type"] == 1:

            reinforce_reward = dvbcq_avg_rew - bcq_avg_rew
            # reinforce_reward = dvbcq_avg_rew

            # reinforce_reward = dvbcq_avg_rew - bcq_avg_rew
            # elif self.dict["moving_average_type"] == 2:

            # reinforce_reward = (dvbcq_avg_rew) - m_avg_v
            # reinforce_reward = 0.0 if ((dvbcq_avg_rew - bcq_avg_rew)) == 0.0 else (1.0/(dvbcq_avg_rew - bcq_avg_rew))
            # reinforce_reward = 1.0 - np.exp(-dvbcq_avg_rew)
            # reinforce_reward = (1.0 - np.exp(-1.0/dvbcq_avg_rew)) - m_avg_v
            #
            # m_avg.add(dvbcq_avg_rew)
            # m_avg.add(1.0 - np.exp(-dvbcq_avg_rew))

            # reinforce_reward_actual = dvbcq_actual_rew - bcq_actual_rew

            # reward_history = np.append(reward_history, dvbcq_avg_rew)
            # if len(reward_history) > 20:
            #     np.delete(reward_history, 0)
            #
            #
            # # Evaluate the updated policy and save evaluations
            # if iteration % self.dict['bl_eval_freq'] == 0:
            #     print(" ")
            #     print(f"BCQ Reward: {bcq_actual_rew}, DVBCQ Reward: {dvbcq_actual_rew}, Normalzied Reward: {reinforce_reward}")
            #     # print(f"BCQ Reward: {bcq_actual_rew}, DVBCQ Reward: {dvbcq_actual_rew}, Normalzied Reward: {reinforce_reward}")
            #     print("Evaluating DVCQL")
            #     perf = self.eval_policy(self.rl_model, eval_episodes=10, seed_num=100, t_env=self.dict["target_env_choice"])
            #     time.sleep(2)


                # if perf > best_perf:
                #     best_perf = perf
                #     mc_perf = perf
                #     self.rl_model.save(self.model_dir, type="best")
                # else:
                #     best_model.load(self.model_dir, type="best")
                #     mc_perf = self.eval_policy(self.rl_model, eval_episodes=10, seed_num=100,
                #                             t_env=self.dict["target_env_choice"])
            #
            #     final_DVBCQ_pef = np.append(final_DVBCQ_pef, perf)
            #     final_DVBCQ_pef_max = np.append(final_DVBCQ_pef_max, mc_perf)
            #
            #     # print("Evaluating Baseline")
            #     # self.eval_policy(self.expert_model, eval_episodes=10, seed_num=100, t_env=self.dict["target_env_choice"])
            # else:
            #     bcq_eval = dvbcq_eval = None
            #
            # bcq_rews = np.append(bcq_rews, bcq_avg_rew)
            # dvbcq_rews = np.append(dvbcq_rews, dvbcq_avg_rew)
            # reinforce_rewards = np.append(reinforce_rewards, reinforce_reward)


            print("----------------------------------------")
            print("Reinforce reward: ", "%.10f" % (reinforce_reward))
            print("----------------------------------------")
            # self.train_dve(self.dve_model, self.dve_optimizer, dvrl_input, self.to_device(sel_prob_curr), reinforce_reward)
            self.train_dve(self.dve_model, self.dve_optimizer, dvrl_input, est_dv_curr, reinforce_reward)

            # self.adaptiveDistanceEpsilon()

            # If a rolling average baseline is being used, then update the rolling avg.
            if self.dict['baseline'] == 'rolling_avg':
                bcq_avg_rew = (self.dict['T'] - 1) * bcq_avg_rew / self.dict['T'] + dvbcq_avg_rew / self.dict['T']
            # #
            # #     # bcq_avg_rew = (self.dict['T'] - 1) * bcq_avg_rew / self.dict['T'] + dvbcq_avg_rew / self.dict['T']
            # #     # bcq_avg_rew = (1.0 - self.dict['tau_baseline']) * bcq_avg_rew + self.dict['tau_baseline'] * dvbcq_avg_rew
            #
            # t_end = time.time()

            # print(f"Iteration:{iteration} took %.2f seconds" % (t_end - t_start))


            # print(f"Iteration:{iteration} took %.3f seconds" % (t_end - t_start))
            # print(f"Sampling and concatenating took %.3f seconds" % (t_sample_concat_end - t_sample_concat_start))
            # print(f"DVE call took %.3f seconds" % (t_dve_end - t_dve_start))
            # print(f"Binomial sampling took %.3f seconds" % (t_binomial_end - t_binomial_start))
            # print(f"Reseting  & Sampling buffer took %.3f seconds" % (t_reseting_samplingBuffer_end - t_reseting_samplingBuffer_start))
            # print(f"Evaluating baseline perf took %.3f seconds" % (t_evaluating_baseline_perf_end - t_evaluating_baseline_perf_start))
            # print(f"Training DVBCQ took %.3f seconds" % (t_training_dvbcq_end - t_training_dvbcq_start))
            # print(f"Evaluating DVBCQ took %.3f seconds" % (t_evaluating_dvbcq_end - t_evaluating_dvbcq_start))
            # print(f"Storing results in NumPy arrays took %.3f seconds" % (t_storing_result_end - t_storing_result_start))
            # print(f"Training DVE took %.3f seconds" % (t_training_dve_end - t_training_dve_start))

            # np.save(f'{self.results_dir}/dvbcq_eval_reward', dvbcq_rews, allow_pickle=True)
            # np.save(f'{self.results_dir}/bcq_eval_reward', bcq_rews, allow_pickle=True)
            # np.save(f'{self.results_dir}/reinforce_reward', reinforce_rewards, allow_pickle=True)
            # np.save(f'{self.results_dir}/est_dv_curr_values', est_dv_curr_values, allow_pickle=True)
            # np.save(f'{self.results_dir}/probs', probs, allow_pickle=True)
            #
            # np.save(f'{self.results_dir}/DVBCQ_avg_reward_{self.dict["target_env_choice"]}', final_DVBCQ_pef,
            #         allow_pickle=True)
            # np.save(f'{self.results_dir}/DVBCQ_avg_reward_max_{self.dict["target_env_choice"]}', final_DVBCQ_pef_max,
            #         allow_pickle=True)



        # Save the RL model
        # self.rl_model.save(self.model_dir, type="final") # Save batch constrained RL model
        self.save_dve(self.model_dir, type="final") # Save data value estimator



    def reset_sampling_buffer(self, states, actions, rewards, next_states, terminals, sel_prob):
        self.sampling_replay_buffer.reset()
        for s, a, r, s_, t, sp in zip(states, actions, rewards, next_states, terminals, sel_prob):
            # if int(sp): # If selected
            # if int(sp) == 1:
            if sp > 0.1:
                self.sampling_replay_buffer.add(s, a, s_, r, 1-t)
                self.ev_buf.add(s, a, s_, r, 1-t)

    def save_dve(self, path, type):
        """
        Save reinforce model
        """
        torch.save(self.dve_model.state_dict(), path + f"_reinforce_{type}")
        torch.save(self.dve_optimizer.state_dict(), path + f"_reinforce_optimizer_{type}")

    def load_dve(self, path, type):
        """
        Load reinforce model
        """
        self.dve_model.load_state_dict(torch.load(path + f"_reinforce_{type}", map_location=torch.device(self.device)))
        self.dve_optimizer.load_state_dict(torch.load(path + f"_reinforce_optimizer_{type}", map_location=torch.device(self.device)))

    def normalized_eval_policy_perStep(self, model, model_name="", eval_episodes=10, seed_num=None):
        if seed_num == None:
            seed_num = 0
        else:
            seed_num = seed_num
        avg_reward = 0.
        episodes_rewards = []
        avg_reward_dict = {}
        for ep_id in range(eval_episodes):
            ep_reward = 0.
            state, done = self.target_env.reset(), False
            self.target_env.seed(seed_num)
            ts_counter = 0
            while not done:
                if ts_counter not in avg_reward_dict:
                    avg_reward_dict[ts_counter] = []
                action = model.select_action(np.array(state))
                state, reward, done, _ = self.target_env.step(action)
                avg_reward_dict[ts_counter].append(reward)
                ts_counter += 1
                ep_reward += reward
            episodes_rewards.append(ep_reward)
            seed_num += 100

        avg_reward_m = [np.mean(i) for i in list(avg_reward_dict.values())]
        episodes_rewards = np.array(avg_reward_m)

        # episodes_rewards = np.array(episodes_rewards)
        # norm_avg_rewards = (episodes_rewards - np.min(episodes_rewards)) / (np.max(episodes_rewards) - np.min(episodes_rewards))
        # episodes_rewards -= np.mean(avg_reward_N)
        # norm_avg_rewards = episodes_rewards/np.max(episodes_rewards)

        avg_reward = np.sum([np.mean(i) for i in list(avg_reward_dict.values())])

        # avg_reward_m -= np.mean(avg_reward_m)
        # norm_avg_reward = np.mean(avg_reward_m)

        episodes_rewards = np.array(episodes_rewards)

        if np.max(episodes_rewards) > self.g_max:
            self.g_max = np.max(episodes_rewards)

        if np.min(episodes_rewards) < self.g_min:
            self.g_min = np.min(episodes_rewards)

        val = (episodes_rewards - self.g_min) / (self.g_max - self.g_min)
        # n_val = val * 0.2 - 0.1
        n_val = val * 2 - 1

        norm_avg_reward = np.mean(n_val)

        print("---------------------------------------")
        print(
            f"{model_name}: Evaluation over {eval_episodes} episodes: Normalized Average Reward {norm_avg_reward:.3f}")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: Average Reward {avg_reward:.3f}")
        print("---------------------------------------")
        return norm_avg_reward, avg_reward

    def eval_policy_KLD(self, target_buffer, data, batch_size=10000, model_name="", mode=""):

        # pdist = nn.PairwiseDistance(p=2).to(self.device)
        kl_v = 0
        rewards_ev = []
        avg_reward = 0.0
        target_buffer.reset_index()
        states, actions, next_states, rewards, dones = target_buffer.sample(batch_size=batch_size, to_device=False)

        # states = detach(states)
        # rewards = detach(rewards)
        # if self.expert_actions is None:
        #     self.expert_actions = torch.FloatTensor(self.expert_model.select_actions(states)).to(self.device)

        # if model_name == "DVBCQ":

        # dvbcq_actions = torch.FloatTensor(model.select_actions(states)).to(self.device)

        # exp_action_dist = pdist(actions, self.expert_actions)
        #
        # dvbcq_action_dist = pdist(actions, dvbcq_actions)
        if mode == "state":
            kl_v = kldiv(data, states)
        elif mode == "action":
            kl_v = kldiv(data, actions)
        elif mode == "state_action":
            kl_v = kldiv(data, np.concatenate((states, actions), axis=1))
        elif mode == "state_action_nextstate":
            kl_v = kldiv(data, np.concatenate((states, actions, next_states), axis=1))

        # elif self.dict['kl_mode'] == "custom1":
        #     kl_v_sasp = kldiv(np.concatenate((states, actions, next_states), axis=1), data)
        #     kl_v_s = kldiv(states, data_c)
        #     kl_v = kl_v_sasp - kl_v_s

        elif mode == "all":
            kl_v = kldiv(data, np.concatenate((states, actions, rewards, next_states, dones), axis=1))

    # rewards.extend(kl_v / float(batch_size))


        print("---------------------------------------")
        # print(f"{model_name}: Evaluation over {batch_size} samples: {kl_v / float(batch_size):.3f}")
        print(f"{model_name}: KL_D: {1.0/kl_v:.8f}")
        # print(f"{model_name}: Evaluation over {batch_size} samples: {np.sum(rewards_ev)/ float(batch_size):.3f}")
        print("---------------------------------------")
        return 1.0/kl_v
        # return np.sum(rewards_ev)/float(batch_size)


    def eval_policy_byAction(self, model, target_buffer, batch_size=10000, model_name="", eps=1.5):

        pdist = nn.PairwiseDistance(p=2).to(self.device)
        num_correct = 0
        rewards_ev = []
        avg_reward = 0.0
        target_buffer.reset_index()
        # for b in range(0, eval_max_size, batch_size):
        # state, action, next_state, reward, done = target_buffer.sample(1, ind=i, to_device=False)
        # state, action, next_state, reward, done = target_buffer.sample(batch_size=batch_size, to_device=False)
        # states, actions, next_states, rewards, dones = target_buffer.sample(batch_size=batch_size, to_device=True)
        states, actions, next_states, rewards, dones = target_buffer.sample(batch_size=batch_size, to_device=True)




        # print("Actions**", action)
        # print("Actions size**", action.size())
        states = detach(states)
        rewards = detach(rewards)
        # states_mean = np.mean(states, axis=0)
        # states_std = np.std(states, axis=0)
        # states = (states - states_mean) / states_std
        if self.expert_actions is None:
            self.expert_actions = torch.FloatTensor(self.expert_model.select_actions(states)).to(self.device)

        if model_name == "DVBCQ":
            # expert_action = torch.FloatTensor(self.bl_train_model.select_actions(states)).to(self.device)
            # print(expert_action)
            # print(expert_action.size())

            dvbcq_actions = torch.FloatTensor(model.select_actions(states)).to(self.device)

            exp_action_dist = pdist(actions, self.expert_actions)

            # print("exp_action_dist", exp_action_dist)

            dvbcq_action_dist = pdist(actions, dvbcq_actions)

            # num_g = torch.logical_and(torch.le(dvbcq_action_dist, exp_action_dist), dvbcq_action_dist < torch.max(exp_action_dist))

            num_g = torch.le(dvbcq_action_dist, exp_action_dist)
            # num_g = dvbcq_action_dist < torch.mean(exp_action_dist)
            # print("dvbcq_action_dist", dvbcq_action_dist)

            # print(f"Min: {torch.min(exp_action_dist)}")
            # print(f"Mean: {torch.mean(exp_action_dist)}")
            # print(f"Max: {torch.max(exp_action_dist)}")

            # num_g = dvbcq_action_dist <= eps
            # num_g = dvbcq_action_dist <= torch.mean(exp_action_dist)

            # s_num_g = torch.where(num_g == True, torch.tensor(1, dtype=torch.int).to(self.device), torch.tensor(0, dtype=torch.int).to(self.device))
            s_num_g = torch.sum(num_g)
            num_correct += s_num_g.item()
            # print(num_correct)

            # num_correct = kldiv(detach(actions), dvbcq_actions)

            # print(reward[num_g.detach().cpu().numpy()].flatten())
            # rewards_ev.extend(rewards[num_g.detach().cpu().numpy()].flatten())

        elif model_name == "BCQ":
            bcq_action_dist = pdist(actions, self.expert_actions)

            num_g = bcq_action_dist <= eps
            s_num_g = torch.sum(num_g)
            num_correct += s_num_g.item()

            # num_correct = kldiv(detach(actions), bcq_actions)

            # print(reward[num_g.detach().cpu().numpy()])
            # rewards_ev.extend(rewards[num_g.detach().cpu().numpy()].flatten())
    # rewards.extend(num_correct / float(batch_size))

        # o_rewards = np.array(rewards_ev)
        # rewards = dvbcq_action_dist.detach().cpu().numpy().flatten()
        # rewards_ev = np.array(rewards_ev)
        # rewards -= np.mean(rewards_ev)
        # rewards /= np.std(rewards_ev)
        print("---------------------------------------")
        print(f"{model_name}: Evaluation over {batch_size} samples: {num_correct / float(batch_size):.3f}")
        # print(f"{model_name}: Evaluation over {batch_size} samples: {np.sum(rewards_ev)/ float(batch_size):.3f}")
        print("---------------------------------------")
        return num_correct / float(batch_size)
        # return np.sum(rewards_ev)/float(batch_size)


    def evaluate_policy_discounted(self, model, model_name="", eval_episodes=10):
        avg_reward = 0.
        all_rewards = []
        gamma = 0.99
        self.target_env.seed(100)
        for _ in range(eval_episodes):
            obs = self.target_env.reset()
            done = False
            cntr = 0
            gamma_t = 1
            while ((not done)):
                action = model.select_action(np.array(obs))
                obs, reward, done, _ = self.target_env.step(action)
                avg_reward += (gamma_t * reward)
                gamma_t = gamma * gamma_t
                cntr += 1
            all_rewards.append(avg_reward)
        avg_reward /= eval_episodes
        for j in range(eval_episodes - 1, 1, -1):
            all_rewards[j] = all_rewards[j] - all_rewards[j - 1]

        all_rewards = np.array(all_rewards)
        std_rewards = np.std(all_rewards)
        median_reward = np.median(all_rewards)
        print("---------------------------------------")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        # return avg_reward, std_rewards, median_reward
        # return np.mean((all_rewards - np.mean(all_rewards)) / np.std(all_rewards))
        return avg_reward


    def eval_policy(self, model, model_name="", eval_episodes=10, seed_num=None, disable_tqdm=True, t_env="target_1"):
        """
        Runs policy for X episodes and returns average reward
        A fixed seed is used for the eval environment
        """

        # self.target_env.seed(100)
        self.target1_env.seed(100)
        self.target2_env.seed(100)
        self.target3_env.seed(100)
        self.target4_env.seed(100)

        if t_env=="target_1":
            env = self.target1_env
        elif t_env=="target_2":
            env = self.target2_env
        elif t_env=="target_3":
            env = self.target3_env
        elif t_env=="target_4":
            env = self.target4_env

        avg_reward = 0.

        for _ in tqdm(range(eval_episodes), disable = disable_tqdm):
            state, done = env.reset(), False
            while not done:
                action = model.select_action(np.array(state),eval=True)
                state, reward, done, _ = env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes
        print("---------------------------------------")
        print(f"{self.env_name} | {model_name}: Evaluation over {eval_episodes} episodes of {t_env}: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def normalized_eval_policy(self, model, model_name="", eval_episodes=10, seed_num=None, disable_tqdm=True, env="target"):
        """
        Runs policy for X episodes and returns average reward
        A fixed seed is used for the eval environment
        """
        if seed_num == None:
            seed_num = 100

        avg_reward = 0.
        episodes_rewards = []

        mean = 0.0
        std = 0.0
        all_scores = []
        scores = []
        gamma = 0.99

        self.target_env.seed(100)
        for _ in tqdm(range(eval_episodes), disable = disable_tqdm):
            ep_reward = 0.
            state, done = self.target_env.reset(), False
            gamma_t = 1
            ep_scores = []
            dis_ep_scores = []

            while not done:
                action = model.select_action(np.array(state))
                state, reward, done, _ = self.target_env.step(action)
                avg_reward += reward

                # ep_reward += (gamma_t * reward)
                ep_reward += reward

                ep_scores.append(ep_reward)
                # ep_scores.append(reward)
                gamma_t = gamma * gamma_t

            # for j in range(eval_episodes - 1, 1, -1):
            #     ep_scores[j] = ep_scores[j] - ep_scores[j - 1]

            # dis_ep_scores.append(np.sum(signal.lfilter([1.0], [1.0, -gamma], ep_scores[::-1])[::-1]))
            # episodes_rewards.append(np.sum(ep_scores))
            episodes_rewards.append(ep_reward)
            # all_scores.append(np.sum(scores))
            # all_scores.append(scores)
            all_scores.extend(scores)
            # all_scores.append(np.sum(discount(r=scores, gamma= 0.99, normal=True)))

            # seed_num+=100

        avg_reward /= eval_episodes






        # episodes_rewards = np.array(all_scores)
        # episodes_rewards = normalize_array(episodes_rewards)

        episodes_rewards = np.array(episodes_rewards)

        if np.max(episodes_rewards) > self.g_max:
            self.g_max = np.max(episodes_rewards)

        if np.min(episodes_rewards) < self.g_min:
            self.g_min = np.min(episodes_rewards)

        val = (episodes_rewards - self.g_min) / (self.g_max - self.g_min)
        # n_val = val * 0.2 - 0.1
        n_val = val * 2 - 1

        # episodes_rewards = standardize_array(episodes_rewards)
        # mean = np.mean(episodes_rewards)
        # std = np.std(episodes_rewards)
        # episodes_rewards -= mean
        # episodes_rewards /= std

        # discounted_return = [np.sum(standardize_array(dr)) for dr in all_scores]
        # norm_avg_reward = np.sum(discounted_return) / eval_episodes

        # norm_avg_reward = np.mean(episodes_rewards)
        norm_avg_reward = np.mean(n_val)

        print("---------------------------------------")
        # print(f"{model_name}: Evaluation over {eval_episodes} episodes: Normalized Average Reward {norm_avg_reward:.3f}")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: Average Reward {avg_reward:.3f}")
        print("---------------------------------------")
        return norm_avg_reward, avg_reward



    def eval_policy_perStep(self, model, model_name="", eval_episodes=10):
        """
        Runs policy for X episodes and returns average reward per time-step
        A fixed seed is used for the eval environment
        """
        avg_reward_dict = {}
        for ep_id in range(eval_episodes):
            state, done = self.target_env.reset(), False
            ts_counter = 0
            while not done:
                if ts_counter not in avg_reward_dict:
                    avg_reward_dict[ts_counter] = []
                action = model.select_action(np.array(state))
                state, reward, done, _ = self.target_env.step(action)
                avg_reward_dict[ts_counter].append(reward)
                ts_counter+=1

        avg_reward = np.sum([np.mean(i) for i in list(avg_reward_dict.values())])
        print("---------------------------------------")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward


    def remove_fraction_eval(self, dataset, data_values, reverse=False, remove=0.1, bcq_num=1):
        """
        Evaluate model trained with a fraction of samples excluded.
        reverse: False results in excluding lowest samples, true excluded highest
        """
        print(f"Constructing Dataset - Excluding High Value {reverse} - Proportion {remove}")
        dataset_tmp = []
        evals_act = np.array([])
        evals_buf = np.array([])

        # print(data_values)
        # print(data_values.shape)

        for i in range(dataset.size):#O
        # for i in range(len(data_values)):
        # gen = dataset.get_batch_generator(1, to_device=False)

        # for i in tqdm(range(0, dataset.size, batch_size)):#O
        # for i, batch in enumerate(tqdm(gen)):
            s, a, s_, r, nd = dataset.sample(1, ind=i, to_device=False)#O
            # s, a, s_, r, nd = batch
            dataset_tmp.append({'s': s, 'a': a, 's_': s_, 'r': r, 'nd': nd, 'v': data_values[i]})
        dataset_tmp = sorted(dataset_tmp, key=lambda k: k['v'], reverse=reverse)

        # Train batch constrained rl model with a dataset where a specified fraction of
        # high/low value samples have been removed
        start_idx = int(len(dataset_tmp) * remove)

        dataset_train = ReplayBuffer(self.state_dim, self.action_dim, self.device)
        for t in dataset_tmp[start_idx:]:
            dataset_train.add(t['s'], t['a'], t['s_'], t['r'], 1. - t['nd'])

            # Load eval model
        # if self.dict['rl_model'] == 'BCQ' and self.dict['expert_path'] != None:
        if self.dict['rl_model'] == 'BCQ':
            eval_model = self.get_bcq()


        print("Training Eval Model")
        for iteration in tqdm(range(0, self.dict['eval_train_iterations'], self.dict["eval_freq"])):

                # _ = eval_model.train(dataset_train,
            eval_model.train(dataset_train,
                             # self.dict['eval_train_iterations'],
                             int(self.dict['eval_freq']),
                             batch_size=self.dict['mini_batch_size'],
                            disable_tqdm=False)

            # Eval on target domain
            samples_removed = 'low value' if reverse == False else 'high value'
            print("Evaluating %s with %f of %s samples removed" % (self.dict['rl_model'], remove, samples_removed))

            # avg_reward = self.eval_policy(eval_model, "DVBCQ_Plot", eval_episodes=100, seed_num=0)

            avg_reward_act = self.eval_policy(eval_model, "DVBCQ_Plot", eval_episodes=100,
                                          seed_num=100, disable_tqdm=False, env="target")
            avg_reward_buf = self.eval_policy_byAction(eval_model, self.target_dataset, model_name="DVBCQ", batch_size=self.dict["eval_batch_size"], eps=self.dist_epsilon)

            evals_act = np.append(evals_act, avg_reward_act)
            evals_buf = np.append(evals_buf, avg_reward_buf)

        np.save(f'{self.results_dir}/eval_freq_evals_act_{self.dict["remove_frac_model"]}_{str(reverse)}_{remove}', evals_act, allow_pickle=True)
        np.save(f'{self.results_dir}/eval_freq_evals_buf_{self.dict["remove_frac_model"]}_{str(reverse)}_{remove}', evals_buf, allow_pickle=True)

        return avg_reward_act, avg_reward_buf

    def remove_fraction_evals(self, dataset, data_values, remove_fraction=np.linspace(0.1, 0.5, 5), model_name="", exclude_high=True):
    # def remove_fraction_evals(self, dataset, data_values, remove_fraction=0.3, model_name="", exclude_high=True):
        """
        Remove a fraction of the highest/lowest value samples and re-train
        a batch constrained RL model from scratch. Result is sotred, so
        we can estimate the extent to which our dve can spot high/low value
        samples. The Batch Constrained Model should perform worse when
        high value samples are excluded, and better when low values samples
        are removed.
        dataset: dataset from which samples are obtained for training the batch constrained model
        data_values: value estimate for each respective sample in the dataset
        remove: list containing the fractions of high/low value samples to be removed.
        """
        file_path = f"{self.results_dir}/{self.dict['env']}_evaluations_{model_name}.csv"
        evals = []

        for r in remove_fraction:
            for exclude_high in [True, False]:
                rew_act, rew_buf = self.remove_fraction_eval(dataset,
                                                              data_values,
                                                              exclude_high,
                                                              remove=r)
                evals.append({'fraction': r,
                              # 'bcq_num': bcq_num,
                              'exclude_high': exclude_high,
                              'avg_reward_act': rew_act,
                              'avg_reward_buf': rew_buf})
                # bcq_num=bcq_num)})
                pd.DataFrame(evals).to_csv(file_path)



    def data_valuate(self, dataset, batch_size):
        """
        Estimate the value of each sample in the specified data set
        """
        print('save data values')
        file_path = '%s/dvrl_%s_train_%d.json' % (self.results_dir, self.dict["env"], len(dataset))
        data_values = []
        sel_vec = []
        not_dones = []


        rng = Generator(PCG64(12345))

        from pathlib import Path

        # rein_best = Path(self.model_dir + "_reinforce_best")
        # if rein_best.is_file():
        #     self.load_dve(self.model_dir, type="final")

        for i in tqdm(range(0, dataset.size, batch_size)):


            # s, a, s_, r, nd = dataset.sample(batch_size, ind=i, to_device=False)#O
            s, a, s_, r, nd = dataset.sample(batch_size=batch_size, to_device=False)#O

            with torch.no_grad():
                batch_values = self.dve_model(concat(s, s_, a, r, nd, self.device))
                # batch_values = self.dve_model(torch.FloatTensor(np.hstack((s, s_, r, nd))).to(self.device))
                # batch_values = self.dve_model(torch.FloatTensor(s).to(self.device))
            data_values.extend(detach(batch_values))
            # sel_vec.extend(rng.binomial(1, detach(batch_values), batch_values.shape))
            not_dones.extend(nd)
        data_values = np.array(data_values)
        # sel_vec = np.array(sel_vec)
        sel_vec = rng.binomial(1, data_values, data_values.shape)

        not_dones = np.array(not_dones)

        print(data_values)


        np.save(f'{self.results_dir}/plot_data_values', data_values, allow_pickle=True)
        np.save(f'{self.results_dir}/plot_not_dones', not_dones, allow_pickle=True)

        dvrl_out = [str(data_values[i]) for i in range(data_values.shape[0])]
        json.dump(dvrl_out, open(file_path, 'w'), indent=4)

        # pd.DataFrame({"values": data_values.flatten(), "labels": labels.flatten()}).to_csv(f"{self.results_dir}/data_values_labels_df.csv")

        return data_values, sel_vec


    def remove_fraction_evals_rf(self, dataset, data_values, remove_fraction=0.3, model_name="", exclude_high=True):

        file_path = f"{self.results_dir}/{self.dict['env']}_evaluations_{model_name}.csv"
        evals = []



        print(f"Constructing Dataset - Excluding High Value {exclude_high} - Proportion {remove_fraction}")
        dataset_tmp = []
        evals = np.array([])


        for i in range(dataset.size):

            s, a, s_, r, nd = dataset.sample(1, ind=i, to_device=False)#O
            # s, a, s_, r, nd = batch
            dataset_tmp.append({'s': s, 'a': a, 's_': s_, 'r': r, 'nd': nd, 'v': data_values[i]})
        dataset_tmp = sorted(dataset_tmp, key=lambda k: k['v'], reverse=exclude_high)

        # Train batch constrained rl model with a dataset where a specified fraction of
        # high/low value samples have been removed
        start_idx = int(len(dataset_tmp) * remove_fraction)

        dataset_train = ReplayBuffer(self.state_dim, self.action_dim, self.device)
        for t in dataset_tmp[start_idx:]:
            dataset_train.add(t['s'], t['a'], t['s_'], t['r'], 1. - t['nd'])

            # Load eval model
        # if self.dict['rl_model'] == 'BCQ' and self.dict['expert_path'] != None:


        print("Training Eval Model")
        for iteration in tqdm(range(0, self.dict['eval_train_iterations'], self.dict["eval_freq"])):

            if self.dict['rl_model'] == 'BCQ':
                if self.dict['remove_frac_model'] == 'fresh':
                    eval_model = self.get_bcq()
                elif self.dict['remove_frac_model'] == 'dvbcq':
                    eval_model = copy.deepcopy(self.rl_model)
                elif self.dict['remove_frac_model'] == 'expert':
                    # eval_model = copy.deepcopy(self.bl_train_model)
                    eval_model = self.get_bcq()
                    eval_model.load(self.dict['expert_path'], type="best")
                # elif self.dict['remove_frac_model'] == 'bl_valid':
                #     eval_model = copy.deepcopy(self.bl_valid_model)
                else:
                    eval_model = self.get_bcq()

            # _ = eval_model.train(dataset_train,
            eval_model.train(dataset_train,
                             # self.dict['eval_train_iterations'],
                             int(self.dict['eval_freq']),
                             batch_size=self.dict['mini_batch_size'],
                            disable_tqdm=False)

            # Eval on target domain
            samples_removed = 'low value' if exclude_high == False else 'high value'
            print("Evaluating %s with %f of %s samples removed" % (self.dict['rl_model'], remove_fraction, samples_removed))


            avg_reward = self.eval_policy(eval_model, "DVBCQ_Plot", eval_episodes=100,
                                          seed_num=2511, disable_tqdm=False, env="target")
            evals = np.append(evals, avg_reward)

        np.save(f'{self.results_dir}/eval_freq_evals', evals, allow_pickle=True)


class MovingAvg:

    # Initializing class
    # Size we will use to compute moving average (100)
    def __init__(self, max_size):
        self.list_of_rewards = []
        self.max_size = max_size
        self.size = 0

    # Add cumulative reward to list of rewards
    def add(self, reward):
        # If rewards is a list, add it to the current list of rewards
        if isinstance(reward, list):
            # self.list_of_rewards += rewards
            self.list_of_rewards.append(reward)
            self.size += 1

        # If rewards is not a list, add reward by append
        else:
            self.list_of_rewards.append(reward)
            self.size += 1

        # Makes sure list remains equal to given size
        while len(self.list_of_rewards) > self.max_size:
            del self.list_of_rewards[0]
            self.size -= 1

    # Compute moving average of list of rewards
    def average(self):
        return np.mean(self.list_of_rewards)

    # Compute moving std of list of rewards
    def std(self):
        return np.std(self.list_of_rewards)
