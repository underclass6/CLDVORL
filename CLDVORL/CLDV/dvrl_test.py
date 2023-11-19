""" Data Valuation based Batch-Constrained Reinforcement Learning """
from itertools import cycle

import scipy.special
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset
from CLDVORL.data_valuation.utils.utils import detach, concat, concat_marginal_information
from CLDVORL.data_valuation.agents.bcq import BCQ
# from CLDVORL.data_valuation.agents.bcq_test import BCQ
from CLDVORL.data_valuation.agents.bear import BEAR
from CLDVORL.data_valuation.agents.TD3_BC import TD3_BC
# from CLDVORL.data_valuation.agents.algos import BEAR
from CLDVORL.data_valuation.agents.cql_sac import CQLSAC
from CLDVORL.data_valuation.utils import utils
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
import random
from scipy.special import kl_div
from CLDVORL.CLDV.delta_classifier import DeltaCla
# from valuation import value_of_data_bcq
import matplotlib.pyplot as plt
from matplotlib import cm

from CLDVORL.CLDV.replay_buffer import ReplayBuffer
from CLDVORL.data_valuation.agents.reinforce import REINFORCE
from CLDVORL.CLDV.utils import get_gym

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
        # self.target_env = target_env
        # self.env_name = self.dict['env']
        self.env_name = args.task

        self.dist_epsilon = self.dict['dist_epsilon']
        self.epsilon_minimum = 1.2
        self.epsilon_decay = 0.99

        # self.make_target_envs(ex_configs)
        self.make_envs()


        self.g_min = 10000.
        self.g_max = 0.

        self.g_dvbcq_rew_history = []
        self.g_bcq_rew_history = []

        self.expert_actions = None

        # if self.dict['rl_model'] == 'BCQ':
        #     self.expert_model = self.get_bcq()
        #     self.rl_model = self.get_bcq()
        #     self.random_model = self.get_bcq()
        # elif self.dict['rl_model'] == 'BEAR':
        #     self.expert_model = self.get_bear()
        #     self.rl_model = self.get_bear()
        #     self.random_model = self.get_bear()
        # elif self.dict['rl_model'] == 'CQL':
        #     self.expert_model = self.get_cql()
        #     self.rl_model = self.get_cql()
        #     self.random_model = self.get_cql()
        # elif self.dict['rl_model'] == 'TD3_BC':
        #     self.expert_model = self.get_td3bc()
        #     self.rl_model = self.get_td3bc()
        #     self.random_model = self.get_td3bc()



        # Used to store samples selected by the value estimator in each outer iteration.
        self.sampling_replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, device)


        self.ev_buf = ReplayBuffer(self.state_dim, self.action_dim, device)

        # Input is a concat of (s, s', a, r, t)
        # reinforce_state_dim = self.state_dim * 2 + self.action_dim + 2
        # Input is a concat of (s, s', a)
        reinforce_state_dim = self.state_dim * 2 + self.action_dim
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

        args.dcla_cuda = True
        self.delta = DeltaCla(target_env.observation_space.shape[0], target_env.action_space.shape[0], args)
        # self.delta.train(rbuf_source, rbuf_target, args_dv)
        # self.delta.save_delta_models('delta')
        # self.delta.load_delta_models('delta')

        self.model = None
        # record average rewards
        self.avg_rewards = []

        self.delta_ratio = args.dcla_ratio

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

    def make_envs(self):
        self.target_env = get_gym(self.dict['d4rl_target_env'])


    def to_device(self, x):
        """
        Copy x to GPU
        """
        return torch.FloatTensor(x).to(self.device)

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
            # model = self.get_bcq()
            # best_model = self.get_bcq()
            if self.model is None:
                model = self.get_bcq()
                best_model = self.get_bcq()
                self.model = model
            else:
                model = self.model
                best_model = self.model
        elif m_type == "BEAR":
            model = self.get_bear()
            best_model = self.get_bear()
        elif m_type == "TD3_BC":
            # model = self.get_td3bc()
            # best_model = self.get_td3bc()
            if self.model is None:
                model = self.get_td3bc()
                best_model = self.get_td3bc()
                self.model = model
            else:
                model = self.model
                best_model = self.model
        elif m_type == "CQL":
            # model = self.get_cql()
            # best_model = self.get_cql()
            if self.model is None:
                model = self.get_cql()
                best_model = self.get_cql()
                self.model = model
            else:
                model = self.model
                best_model = self.model


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

            # r = self.eval_policy(model, eval_episodes=eval_episodes, t_env=self.dict["target_env_choice"])
            r = self.eval_policy_norm_score(model, eval_episodes=eval_episodes, t_env=self.dict["target_env_choice"])


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
                elif self.dict['rl_model'] == 'TD3_BC':
                    best_model.load(model_path, type="best")
                elif self.dict['rl_model'] == 'CQL':
                    best_model.load(model_path, type="best")
                # c_max_reward = self.eval_policy(best_model, eval_episodes=eval_episodes, t_env=self.dict["target_env_choice"])
                c_max_reward = self.eval_policy_norm_score(best_model, eval_episodes=eval_episodes, t_env=self.dict["target_env_choice"])



            bl_avg_reward = np.append(bl_avg_reward, r)
            bl_avg_reward_max = np.append(bl_avg_reward_max, c_max_reward)

            # record average rewards
            self.avg_rewards.append(r)

            training_iters += int(self.dict['bl_eval_freq'])

            print(f"Env: {self.dict['d4rl_target_env']} | Training iterations: {training_iters} | Avg Reward: {r}")


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

        # if self.dict['rl_model'] == "BCQ":
        #     best_model = self.get_bcq()
        # elif self.dict['rl_model'] == "BEAR":
        #     best_model = self.get_bear()
        # elif self.dict['rl_model'] == "CQL":
        #     best_model = self.get_cql()
        # elif self.dict['rl_model'] == "TD3_BC":
        #     best_model = self.get_td3bc()


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

        # ---------------------------------------------from zekun-------------------------------------------------------
        dv_avg_rew = 0.0
        # ---------------------------------------------from zekun-------------------------------------------------------

        # org_bcq_avg_rew = self.eval_policy(self.expert_model, eval_episodes=10, seed_num=100, t_env=self.dict["target_env_choice"])
        # reward_history = np.append(reward_history, dvbcq_avg_rew)

        self.expert_perf = bcq_actual_rew
        update_dve = True

        # m_avg = MovingAvg(20)
        # m_avg_v = 0.0

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


            # dvrl_input = concat(states, next_states, actions, rewards, terminals, self.device)
            dvrl_input = torch.FloatTensor(np.hstack((states,
                                             next_states,
                                             actions))).to(self.device)


            est_dv_curr = self.dve_model(dvrl_input)


            # reinforce_reward = dvbcq_avg_rew - bcq_avg_rew

            # --------------------------------------baseline1-----------------------------------------------------------
            # delta_avg = -np.sum(est_dv_curr.squeeze(1).detach().cpu().numpy() * np.abs(self.delta.delta_dynamic(torch.cuda.FloatTensor(states), torch.cuda.FloatTensor(actions), torch.cuda.FloatTensor(next_states))))
            ratio = self.delta_ratio
            delta_avg = ratio*-np.sum(est_dv_curr.squeeze(1).detach().cpu().numpy() * np.abs(self.delta.delta_dynamic(torch.cuda.FloatTensor(states), torch.cuda.FloatTensor(actions), torch.cuda.FloatTensor(next_states)))) + (1-ratio)*np.sum(est_dv_curr.squeeze(1).detach().cpu().numpy())

            reinforce_reward = delta_avg
            # reinforce_reward = dvbcq_avg_rew
            reinforce_rewards = np.append(reinforce_rewards, reinforce_reward)
            if np.max(reinforce_rewards - np.min(reinforce_rewards)) > 1e-5:
                normalized_reinforce_rewards = 2 * (reinforce_rewards - np.min(reinforce_rewards)) / np.max(reinforce_rewards - np.min(reinforce_rewards)) - 1
            else:
                normalized_reinforce_rewards = reinforce_rewards
            reinforce_reward = normalized_reinforce_rewards[-1]



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
        torch.save(self.dve_model.state_dict(), path + f"reinforce_{type}")
        torch.save(self.dve_optimizer.state_dict(), path + f"reinforce_optimizer_{type}")

    def load_dve(self, path, type):
        """
        Load reinforce model
        """
        self.dve_model.load_state_dict(torch.load(path + f"reinforce_{type}", map_location=torch.device(self.device)))
        self.dve_optimizer.load_state_dict(torch.load(path + f"reinforce_optimizer_{type}", map_location=torch.device(self.device)))



    def eval_policy(self, policy, env, eval_episodes=10, plot=False):
        avg_reward = 0.
        plt.clf()
        start_states = []
        color_list = cm.rainbow(np.linspace(0, 1, eval_episodes + 2))

        for i in range(eval_episodes):
            state, done = env.reset(), False
            states_list = []
            start_states.append(state)
            while not done:
                action = policy.select_action(np.array(state))
                state, reward, done, _ = env.step(action)
                avg_reward += reward
                states_list.append(state)
            states_list = np.array(states_list)

            if plot:
                plt.scatter(states_list[:, 0], states_list[:, 1], color=color_list[i], alpha=0.1)
                plt.scatter(8, 10, color='white', alpha=0.1)
                plt.scatter(2, 0, color='white', alpha=0.1)
        if plot:
            start_states = np.array(start_states)
            plt.scatter(start_states[:, 0], start_states[:, 1], color='red')
            plt.savefig('./eval_fig')

        avg_reward /= eval_episodes
        normalized_score = env.get_normalized_score(avg_reward)

        info = {'AverageReturn': avg_reward, 'NormReturn': normalized_score}
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, {normalized_score:.3f}")
        print("---------------------------------------")
        return normalized_score


    ###### for D4RL
    def eval_policy_norm_score(self, model, model_name="", eval_episodes=10, seed_num=None, disable_tqdm=True, t_env="target_1"):
        """
        Runs policy for X episodes and returns average normalized score
        A fixed seed is used for the eval environment
        """
        env = self.target_env
        env.seed(100)

        sum_reward = 0.

        for _ in tqdm(range(eval_episodes), disable = disable_tqdm):
            state, done = env.reset(), False
            while not done:
                action = model.select_action(np.array(state),eval=True)
                state, reward, done, _ = env.step(action)
                sum_reward += reward
            score = env.get_normalized_score(sum_reward)

        avg_score = score / eval_episodes * 100
        print("---------------------------------------")
        print(f"{self.env_name} | {model_name}: Evaluation over {eval_episodes} episodes of {t_env}: {avg_score:.3f}")
        print("---------------------------------------")
        return avg_score



    # def remove_fraction_evals(self, dataset, data_values, remove_fraction=np.linspace(0.1, 0.5, 5), model_name="", exclude_high=True):
    # # def remove_fraction_evals(self, dataset, data_values, remove_fraction=0.3, model_name="", exclude_high=True):
    #     """
    #     Remove a fraction of the highest/lowest value samples and re-train
    #     a batch constrained RL model from scratch. Result is sotred, so
    #     we can estimate the extent to which our dve can spot high/low value
    #     samples. The Batch Constrained Model should perform worse when
    #     high value samples are excluded, and better when low values samples
    #     are removed.
    #     dataset: dataset from which samples are obtained for training the batch constrained model
    #     data_values: value estimate for each respective sample in the dataset
    #     remove: list containing the fractions of high/low value samples to be removed.
    #     """
    #     file_path = f"{self.results_dir}/{self.dict['env']}_evaluations_{model_name}.csv"
    #     evals = []
    #
    #     for r in remove_fraction:
    #         for exclude_high in [True, False]:
    #             rew_act, rew_buf = self.remove_fraction_eval(dataset,
    #                                                           data_values,
    #                                                           exclude_high,
    #                                                           remove=r)
    #             evals.append({'fraction': r,
    #                           # 'bcq_num': bcq_num,
    #                           'exclude_high': exclude_high,
    #                           'avg_reward_act': rew_act,
    #                           'avg_reward_buf': rew_buf})
    #             # bcq_num=bcq_num)})
    #             pd.DataFrame(evals).to_csv(file_path)



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
                # batch_values = self.dve_model(concat(s, s_, a, r, nd, self.device))
                batch_values = self.dve_model(torch.FloatTensor(np.hstack((s, s_, a))).to(self.device))
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
        print(np.mean(data_values))


        np.save(f'{self.results_dir}/plot_data_values', data_values, allow_pickle=True)
        np.save(f'{self.results_dir}/plot_not_dones', not_dones, allow_pickle=True)

        dvrl_out = [str(data_values[i]) for i in range(data_values.shape[0])]
        json.dump(dvrl_out, open(file_path, 'w'), indent=4)

        # pd.DataFrame({"values": data_values.flatten(), "labels": labels.flatten()}).to_csv(f"{self.results_dir}/data_values_labels_df.csv")

        return data_values, sel_vec

