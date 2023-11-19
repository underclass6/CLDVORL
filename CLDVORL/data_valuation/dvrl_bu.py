""" Data Valuation based Batch-Constrained Reinforcement Learning """
from itertools import cycle

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from data_valuation.utils.utils import detach, concat, concat_marginal_information
from data_valuation.agents.bcq import BCQ
from data_valuation.agents.bear import BEAR, GaussianPolicy
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
import random

from data_valuation.utils.replay_buffer import ReplayBuffer
from data_valuation.agents.reinforce import REINFORCE



class DVRL(object):
    def __init__(self, train_dataset, valid_dataset, device, env, results_dir, args={}):

        self.dict = vars(args)
        self.args = args
        assert len(self.dict) > 0

        # data
        self.model_dir = utils.make_dvrl_dir(results_dir, args)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_space = env.action_space
        self.train_dataset = train_dataset
        self.target_dataset = valid_dataset
        self.results_dir = results_dir
        self.device = device
        self.target_env = env

        self.g_dvbcq_rew_history = []
        self.g_bcq_rew_history = []

        self.bl_train_model = self.get_bcq()
        self.bl_valid_model = self.get_bcq()
        self.rl_model = self.get_bcq()

        self.source_env = utils.get_gym(self.dict["env"], self.dict["source_env_friction"], self.dict["source_env_mass_torso"])

        # Used to store samples selected by the value estimator in each outer iteration.
        self.sampling_replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, device)

        # # models
        # if self.dict['marginal_info']:
        #     # Since input is a concat of (s, s', a, r, t, s'_predict-s, a-a_predict, r-r_predicts, t-t_predict)
        #     reinforce_state_dim = self.state_dim * 3 + self.action_dim + 4
        # else:


        # Input is a concat of (s, s', a, r, t)
        reinforce_state_dim = self.state_dim * 2 + self.action_dim + 2
        reinforce_action_dim = 1 # Outputs probability for each sample
        self.dve_model = REINFORCE(reinforce_state_dim,
                                   reinforce_action_dim,
                                   layers=self.dict['reinforce_layers'],
                                   args=self.dict).to(device)


        # optimizers
        self.dve_optimizer = optim.Adam(self.dve_model.parameters(), self.dict['dve_lr'])

        self.scaler = torch.cuda.amp.GradScaler()

        self.final_buff_train =  self.get_final_transitions(train_dataset)
        self.final_buff_valid =  self.get_final_transitions(valid_dataset)

    def get_final_transitions(self, dataset):
        rbuf = ReplayBuffer(self.target_env.observation_space.shape[0], self.target_env.action_space.shape[0], self.device)

        states, actions, next_states, rewards, terminals = dataset.sample(len(dataset), to_device=False) #O
        # states, actions, next_states, rewards, terminals = next(generator)


        i = 0
        for s, a, r, s_, t in zip(states, actions, rewards, next_states, terminals):
            if (i == 1000):
                done = 1-t
                rbuf.add(s, a, s_, r, done)
                i = 0
            i+=1
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


    def train_dve(self, x, s_input, reward):
        """
         Training data value estimator
         s_input: selection probability
         x_input: Sample tuple
         reward_input: reward signal for training the reinforce agent
        """
        est_data_value = self.dve_model(x)
        self.dve_optimizer.zero_grad()

        # with torch.cuda.amp.autocast():
        prob = torch.sum(s_input * torch.log(est_data_value + self.dict['epsilon']) +\
            (1 - s_input) * torch.log(1 - est_data_value + self.dict['epsilon']))

        loss = (-reward * prob) + 1e3 * (utils.get_maximum(torch.mean(est_data_value)-self.dict['threshold'], 0) +
                   utils.get_maximum(1-torch.mean(est_data_value)-self.dict['threshold'], 0))

        loss.backward()
        self.dve_optimizer.step()

        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.dve_optimizer)
        # self.scaler.update()


    def to_device(self, x):
        """
        Copy x to GPU
        """
        return torch.FloatTensor(x).to(self.device)


    def load_train_baselines(self):
        if self.dict['rl_model'] == 'BCQ':
            if self.dict['trained_train_baseline_path'] != None:
                print('loading train-baseline model.')
                self.bl_train_model.load(self.dict['trained_train_baseline_path'])
            else:
                print('start training baseline model.')
                self.train_baseline(data="train")

            if self.dict['trained_valid_baseline_path'] != None:
                print('loading valid-baseline model.')
                self.bl_valid_model.load(self.dict['trained_valid_baseline_path'])
            else:
                print('start training valid-baseline model.')
                self.train_baseline("valid")

    def load_train_dvbcq(self):
        if self.dict['rl_model'] == 'BCQ':
            if self.dict['rl_model_t'] == 'fresh':
                self.rl_model = self.get_bcq()
            elif self.dict['rl_model_t'] == 'dvbcq' and self.dict['trained_dvbcq_path'] != None:
                self.rl_model = self.get_bcq()
                self.rl_model.load(self.dict['trained_dvbcq_path'])
            elif self.dict['rl_model_t'] == 'bl_train':
                self.rl_model = copy.deepcopy(self.bl_train_model)
            elif self.dict['rl_model_t'] == 'bl_valid':
                self.rl_model = copy.deepcopy(self.bl_valid_model)
            else:
                self.rl_model = self.get_bcq()


    def train_baseline(self, data="train"):

        bl_avg_reward = np.array([])

        training_iters = 0
        bl_path = f"{self.model_dir}/basline_model/{data}/"
        Path(bl_path).mkdir(parents=True, exist_ok=True)
        values = []
        keys = []
        if data == "train":
            states, _, _, rewards, _ = self.final_buff_train.sample(self.final_buff_train.size)
            model = self.bl_train_model
            buffer = self.train_dataset
        elif data == "valid":
            states, _, _, rewards, _ = self.final_buff_valid.sample(self.final_buff_valid.size)
            model = self.bl_valid_model
            buffer = self.target_dataset

        for r in rewards:
            keys.append(str(detach(r)[0]))


        while training_iters < self.dict['bl_training_max_timesteps']:
        # for i in tqdm(range(0, dataset.size, batch_size))
            if self.dict['rl_model'] == 'BCQ':
                # actor_loss, critic_loss, vae_loss = model.train(
                model.train(replay_buffer=buffer,
                            iterations=int(self.dict['eval_freq']),
                            batch_size=self.dict['mini_batch_size'],
                            disable_tqdm=False)


            # r = self.eval_policy(model)
            r, actual_reward = self.normalized_eval_policy(model)
            # r = self.normalized_eval_policy_perStep(model)

            training_iters += int(self.dict['eval_freq'])

            print(f"Training iterations: {training_iters},\
                    Avg Reward: {actual_reward}")
                    # Actor Loss: {actor_loss}\
                    # Critic Loss: {critic_loss}\
                    # VAE Loss: {vae_loss}")

            np.save(f'{self.results_dir}/bl_avg_reward', bl_avg_reward, allow_pickle=True)

            estimates = {}

            value_estimates = model.get_value_estimate(states)

            for key, v  in zip(keys, value_estimates):
                estimates.update({key:v[0]})
            values.append(estimates)

            np.save(f'{self.results_dir}/bl_values',  np.array(values), allow_pickle=True)

        model.save(bl_path)
        print(f"{data}-Basline Model Path: {bl_path}")


    def train(self):
        """
        RL model and DVE iterative training
        """

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



        # self.load_train_baselines()
        self.load_train_dvbcq()
        bl_model = self.get_bcq()
        # bcq_avg_rew = self.eval_policy(bl_model, model_name="BCQ", eval_episodes=10, seed_num=60)
        bcq_avg_rew = self.eval_policy_perStep(bl_model, model_name="BCQ", eval_episodes=10, seed_num=60)
        # bcq_avg_rew, bcq_actual_rew = self.normalized_eval_policy(self.bl_valid_model, model_name="BCQ", eval_episodes=100, seed_num=60)
        # bcq_avg_rew, bcq_actual_rew = self.normalized_eval_policy_perStep(self.bl_train_model, eval_episodes=100, seed_num=60)


        # Get batch constrained reinforcement learning model
        # (currently BCQ is the only option, more to be added)
        # if self.dict['rl_model'] == 'BCQ' and self.dict['trained_train_baseline_path'] != None:



        flag_update = False
        best_reward = 0.0

        # start training
        for iteration in tqdm(range(self.dict['outer_iterations'])):
            seed_num = random.randint(0,1000)
            t_start = time.time()
            # batch select
            # states, actions, next_states, rewards, terminals = self.train_dataset.sample(self.dict['batch_size'], seed=iteration, to_device=False)

            states, actions, next_states, rewards, terminals = self.train_dataset.sample(self.dict['batch_size'], to_device=False) #O


            dvrl_input = concat(states, next_states, actions, rewards, terminals, self.device)

            est_dv_curr = self.dve_model(dvrl_input)

            if self.dict["sampler_dist"] == "binomial":
                # Samples the selection probability
                sel_prob_curr = rng.binomial(1, detach(est_dv_curr), est_dv_curr.shape)

                # Exception (When selection probability is 0)
                if np.sum(sel_prob_curr) == 0:
                    est_dv_curr = 0.5 * np.ones(np.shape(detach(est_dv_curr)))
                    # sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)
                    sel_prob_curr = rng.binomial(1, est_dv_curr, est_dv_curr.shape)
                    est_dv_curr = self.to_device(est_dv_curr)


                est_dv_curr_values = np.append(est_dv_curr_values, detach(torch.as_tensor(est_dv_curr)).flatten())
                probs = np.append(probs, sel_prob_curr.flatten())

                print(detach(torch.as_tensor(est_dv_curr)).flatten())
                print(sel_prob_curr.flatten())

            elif self.dict["sampler_dist"] == "bernoulli":


                sel_prob_curr = torch.bernoulli(est_dv_curr)

                if torch.sum(sel_prob_curr) == 0:
                    # exception when selection probability is 0
                    estimated_dv = 0.5 * torch.ones_like(est_dv_curr)
                    sel_prob_curr = torch.bernoulli(estimated_dv)


                est_dv_curr_values = np.append(est_dv_curr_values, detach(torch.as_tensor(est_dv_curr)).flatten())
                probs = np.append(probs, detach(sel_prob_curr).flatten())


                print(detach(torch.as_tensor(est_dv_curr)).flatten())
                print(detach(sel_prob_curr).flatten())

            # Reset (empty) the sampling buffer, and then add a samples based on
            # the selection probabilities.

            # self.reset_sampling_buffer(states, actions, rewards, next_states, terminals, sel_prob_curr)
            if self.dict["sampler_dist"] == "binomial":
                self.reset_sampling_buffer(states, actions, rewards, next_states, terminals, sel_prob_curr.flatten())
            elif self.dict["sampler_dist"] == "bernoulli":
                self.reset_sampling_buffer(states, actions, rewards, next_states, terminals, detach(sel_prob_curr).flatten())

            # train rl_model with sampled batch data for inner_iterations
            #print("Training Selective Model")
            if self.dict['rl_model'] == 'BCQ':

                # bcq_avg_rew, bcq_actual_rew = self.normalized_eval_policy(self.bl_train_model, "BCQ", eval_episodes=3, seed_num=seed_num)
                # rand_avg_rew, rand_actual_rew = self.normalized_eval_policy(self.random_model, "Random-BCQ", eval_episodes=3, seed_num=seed_num)

                # self.random_model = copy.deepcopy(self.rl_model)
                # _ = self.rl_model.train(replay_buffer=self.sampling_replay_buffer,
                self.rl_model.train(replay_buffer=self.sampling_replay_buffer,
                                        iterations=self.dict['inner_iterations'],
                                        batch_size=min(self.dict['mini_batch_size'],
                                         self.sampling_replay_buffer.size))


                # dvbcq_avg_rew = self.eval_policy(self.rl_model, "DVBCQ", eval_episodes=3, seed_num=seed_num)
                dvbcq_avg_rew = self.eval_policy_perStep(self.rl_model, "DVBCQ", eval_episodes=3, seed_num=seed_num)
                # dvbcq_avg_rew, dvbcq_actual_rew = self.normalized_eval_policy_perStep(self.rl_model, "DVBCQ", eval_episodes=5, seed_num=seed_num)


            # Compute reward for the Reinforce agent.
                # reinforce_reward = bl_critic_loss - critic_loss
                reinforce_reward = dvbcq_avg_rew - bcq_avg_rew

                # if reinforce_reward > 0:
                reward_signal_history = np.append(reward_signal_history, reinforce_reward)
                rew_deque.append(reinforce_reward)
                rl_model_rew_history = np.append(rl_model_rew_history, dvbcq_avg_rew)

                if len(rl_model_rew_history) > 20:
                    rl_model_rew_history = np.delete(rl_model_rew_history, 0)

                # if self.dict['baseline'] == 'rolling_avg':
                #     bl_critic_loss = (1.0 - self.dict['tau']) * bl_critic_loss + self.dict['tau'] * critic_loss

                # Evaluate the updated policy and save evaluations

                if iteration % self.dict['print_eval'] == 0:
                    print(" ")
                    # print(f"Baseline: {bl_critic_loss}, Mean Loss: {critic_loss}, Reinforce Reward: {reinforce_reward}")
                    print(
                        f"BCQ Reward: {bcq_avg_rew}, DVBCQ Reward: {dvbcq_avg_rew}, Reinforce Reward: {reinforce_reward}")
                    print("Evaluating DVBCQ")
                    dvbcq_eval = self.eval_policy(self.rl_model)
                    # dvbcq_eval = self.eval_policy_perStep(self.rl_model)
                    print("Evaluating BCQ")
                    bcq_eval = self.eval_policy(self.bl_train_model)
                    # bcq_eval = self.eval_policy_perStep(self.bl_model)
                else:
                    dvbcq_eval = bcq_eval = None

                bcq_evals = np.append(bcq_evals, bcq_eval)
                dvbcq_evals = np.append(dvbcq_evals, dvbcq_eval)
                bcq_rews = np.append(bcq_rews, bcq_avg_rew)
                dvbcq_rews = np.append(dvbcq_rews, dvbcq_avg_rew)
                # baseline_losses = np.append(baseline_losses, bl_critic_loss)
                # mean_losses = np.append(mean_losses, critic_loss)
                reinforce_rewards = np.append(reinforce_rewards, reinforce_reward)

                # If a rolling average baseline is being used, then update the rolling avg.
                # if self.dict['baseline'] == 'rolling_avg':
                #     baseline_loss = (1.0 - self.dict['tau']) * baseline_loss + self.dict['tau'] * mean_loss
                #     moving_avg = (1.0 - 0.01) * moving_avg + 0.01 * mean_loss

                # Optimize the data value estimator
                t_dve_optimize_start = time.time()

                if self.dict["sampler_dist"] == "binomial":
                    check = np.max(detach(est_dv_curr).flatten()) == np.min(detach(est_dv_curr).flatten())
                    lower_bound_check = np.mean(detach(est_dv_curr).flatten()) >= 0.1
                    upper_bound_check = np.mean(detach(est_dv_curr).flatten()) <= 0.9
                elif self.dict["sampler_dist"] == "bernoulli":
                    check = torch.max(est_dv_curr) == torch.min(est_dv_curr)
                    lower_bound_check = (est_dv_curr * 1.0).mean().item() >= 0.1
                    upper_bound_check = (est_dv_curr * 1.0).mean().item() <= 0.9

                if len(rl_model_rew_history) > 2:


                    max_rew = np.max(rl_model_rew_history.reshape(-1, 1)[:-1])
                    update_flag = True
                    if reinforce_reward > max_rew:
                        best_reward_sig = reinforce_reward



                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    scaler.fit(rl_model_rew_history.reshape(-1, 1))
                    normalized_reward = scaler.transform(rl_model_rew_history.reshape(-1, 1))[-1][0]

                    # normalized_reward = (reward_signal_history - reward_signal_history.mean()) / (reward_signal_history.std() + 0.005)
                    # normalized_reward = normalized_reward.reshape(-1, 1)[-1][0]
                else:
                    update_flag = False
                #     best_reward_sig = 0.0
                #     normalized_reward = 1.0

                # if reinforce_reward > best_reward_sig:
                if update_flag:
                    print(" ------ Normalized Reward -----")
                    print(normalized_reward)
                    print(" ------------------------------")
                    # if not check and update_flag:

                    if self.dict["sampler_dist"] == "binomial":
                        self.train_dve(dvrl_input, self.to_device(sel_prob_curr), normalized_reward)
                    # self.train_dve(dvrl_input, sel_prob_curr, reinforce_reward)
                    elif self.dict["sampler_dist"] == "bernoulli":
                        self.train_dve(dvrl_input, sel_prob_curr, normalized_reward)

                    bcq_avg_rew = (self.dict['T'] - 1) * bcq_avg_rew / self.dict['T'] + dvbcq_avg_rew / self.dict['T']
                    # bcq_avg_rew = (1.0 - self.dict['tau']) * bcq_avg_rew + self.dict['tau'] * dvbcq_avg_rew

                t_dve_optimize_end = time.time()

            # If a rolling average baseline is being used, then update the rolling avg.
            # if self.dict['baseline'] == 'rolling_avg':
            #     bcq_avg_rew = (self.dict['T'] - 1) * bcq_avg_rew / self.dict['T'] + dvbcq_avg_rew / self.dict['T']
            t_end = time.time()
            print(f"Iteration:{iteration} took %.2f seconds" % (t_end - t_start))


        np.save(f'{self.results_dir}/dvbcq_evals', dvbcq_evals, allow_pickle=True)
        np.save(f'{self.results_dir}/bcq_evals', bcq_evals, allow_pickle=True)
        np.save(f'{self.results_dir}/dvbcq_eval_reward', dvbcq_rews, allow_pickle=True)
        np.save(f'{self.results_dir}/bcq_eval_reward', bcq_rews, allow_pickle=True)
        np.save(f'{self.results_dir}/reinforce_reward', reinforce_rewards, allow_pickle=True)
        np.save(f'{self.results_dir}/est_dv_curr_values', est_dv_curr_values, allow_pickle=True)
        np.save(f'{self.results_dir}/probs', probs, allow_pickle=True)


        # Save the RL model
        self.rl_model.save(self.model_dir) # Save batch constrained RL model
        self.save_dve() # Save data value estimator



    def reset_sampling_buffer(self, states, actions, rewards, next_states, terminals, sel_prob_curr):
        self.sampling_replay_buffer.reset()
        for s, a, r, s_, t, sp in zip(states, actions, rewards, next_states, terminals, sel_prob_curr):

            if int(sp): # If selected
                self.sampling_replay_buffer.add(s, a, s_, r, 1-t)

    def save_dve(self):
        """
        Save reinforce model
        """
        torch.save(self.dve_model.state_dict(), self.model_dir + "_reinforce")
        torch.save(self.dve_optimizer.state_dict(), self.model_dir + "_reinforce_optimizer")

    def load_dve(self, filename):
        """
        Load reinforce model
        """
        self.dve_model.load_state_dict(torch.load(filename + "_reinforce", map_location=torch.device(self.device)))
        self.dve_optimizer.load_state_dict(torch.load(filename + "_reinforce_optimizer", map_location=torch.device(self.device)))

    def eval_policy(self, model, model_name="", eval_episodes=10, seed_num=None, disable_tqdm=True, env="target"):
        """
        Runs policy for X episodes and returns average reward
        A fixed seed is used for the eval environment
        """
        if seed_num == None:
            seed_num = 0
        else:
            seed_num = seed_num
        avg_reward = 0.
        for _ in tqdm(range(eval_episodes), disable = disable_tqdm):
            if env == "target":
                state, done = self.target_env.reset(), False
            elif env == "source":
                state, done = self.source_env.reset(), False
            self.target_env.seed(seed_num)
            while not done:
                action = model.select_action(np.array(state))
                state, reward, done, _ = self.target_env.step(action)
                avg_reward += reward
            seed_num+=100
        avg_reward /= eval_episodes
        print("---------------------------------------")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def normalized_eval_policy(self, model, model_name="", eval_episodes=10, seed_num=None, disable_tqdm=True, env="target"):
        """
        Runs policy for X episodes and returns average reward
        A fixed seed is used for the eval environment
        """
        if seed_num == None:
            seed_num = 0
        else:
            seed_num = seed_num
        avg_reward = 0.
        episodes_rewards = []

        mean = 0.0
        std = 0.0

        for _ in tqdm(range(eval_episodes), disable = disable_tqdm):
            ep_reward = 0.
            state, done = self.target_env.reset(), False
            self.target_env.seed(seed_num)

            while not done:
                action = model.select_action(np.array(state))
                state, reward, done, _ = self.target_env.step(action)
                avg_reward += reward
                ep_reward += reward
            episodes_rewards.append(ep_reward)
            seed_num+=100

        avg_reward /= eval_episodes

        episodes_rewards = np.array(episodes_rewards)

        # if model_name == "DVBCQ":
        #     self.g_dvbcq_rew_history = np.append(self.g_dvbcq_rew_history, episodes_rewards)
        #     mean = np.mean(self.g_dvbcq_rew_history)
        #     std = np.std(self.g_dvbcq_rew_history)
        # elif model_name == "BCQ":
        #     self.g_bcq_rew_history = np.append(self.g_bcq_rew_history, episodes_rewards)
        #     mean = np.mean(self.g_bcq_rew_history)
        #     std = np.std(self.g_bcq_rew_history)
        # else:
        #     print("!!!")


        mean = np.mean(episodes_rewards)
        std = np.std(episodes_rewards)
        episodes_rewards -= mean
        episodes_rewards /= std

        episodes_rewards = -1 * (episodes_rewards - np.min(episodes_rewards)) / np.ptp(episodes_rewards) - 1
        # norm_avg_reward = (episodes_rewards - np.mean(episodes_rewards)) / (np.std(episodes_rewards) + 1e-10)
        norm_avg_reward = np.mean(episodes_rewards)

        print("---------------------------------------")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: Normalized Average Reward {norm_avg_reward:.3f}")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: Average Reward {avg_reward:.3f}")
        print("---------------------------------------")
        return norm_avg_reward, avg_reward


    def eval_policy_perStep(self, model, model_name="", seed_num=None, eval_episodes=10):
        """
        Runs policy for X episodes and returns average reward per time-step
        A fixed seed is used for the eval environment
        """
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

        avg_reward = np.sum([np.mean(i) for i in list(avg_reward_dict.values())])
        print("---------------------------------------")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward



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
                ts_counter+=1
                ep_reward += reward
            episodes_rewards.append(ep_reward)
            seed_num += 100

        avg_reward_m = [np.mean(i) for i in list(avg_reward_dict.values())]
        avg_reward_m = np.array(avg_reward_m)

        # episodes_rewards = np.array(episodes_rewards)
        # norm_avg_rewards = (episodes_rewards - np.min(episodes_rewards)) / (np.max(episodes_rewards) - np.min(episodes_rewards))
        # episodes_rewards -= np.mean(avg_reward_N)
        # norm_avg_rewards = episodes_rewards/np.max(episodes_rewards)

        avg_reward = np.sum([np.mean(i) for i in list(avg_reward_dict.values())])


        episodes_rewards -= np.mean(avg_reward_m)
        avg_reward_m /= np.std(episodes_rewards)
        norm_avg_reward = np.mean(avg_reward_m)

        print("---------------------------------------")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: Normalized Average Reward {norm_avg_reward:.3f}")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: Average Reward {avg_reward:.3f}")
        print("---------------------------------------")
        return norm_avg_reward, avg_reward

    def remove_fraction_eval(self, dataset, data_values, reverse=False, remove=0.1, bcq_num=1):
        """
        Evaluate model trained with a fraction of samples excluded.
        reverse: False results in excluding lowest samples, true excluded highest
        """
        print(f"Constructing Dataset - Excluding High Value {reverse} - Proportion {remove}")
        dataset_tmp = []

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
            dataset_train.add(t['s'], t['a'], t['s_'], t['r'], t['nd'])

            # Load eval model
        # if self.dict['rl_model'] == 'BCQ' and self.dict['trained_train_baseline_path'] != None:
        if self.dict['rl_model'] == 'BCQ':
            if self.dict['remove_frac_model'] == 'fresh':
                eval_model = self.get_bcq()
            elif self.dict['remove_frac_model'] == 'dvbcq':
                eval_model = copy.deepcopy(self.rl_model)
            elif self.dict['remove_frac_model'] == 'bl_train':
                eval_model = copy.deepcopy(self.bl_train_model)
            elif self.dict['remove_frac_model'] == 'bl_valid':
                eval_model = copy.deepcopy(self.bl_valid_model)
            else:
                eval_model = self.get_bcq()

        print("Training Eval Model")
        _ = eval_model.train(dataset_train,
                             self.dict['eval_train_iterations'],
                             batch_size=self.dict['mini_batch_size'])

        # Eval on target domain
        samples_removed = 'low value' if reverse == False else 'high value'
        print("Evaluating %s with %f of %s samples removed" % (self.dict['rl_model'], remove, samples_removed))

        # avg_reward = self.eval_policy(eval_model, "DVBCQ_Plot", eval_episodes=100, seed_num=0)

        avg_reward = self.eval_policy(eval_model, f"DVBCQ_Plot for BCQ num: {bcq_num}", eval_episodes=100,
                                      seed_num=2511, disable_tqdm=False, env="target")

        return avg_reward

    def remove_fraction_evals(self, dataset, data_values, remove=np.linspace(0.1, 0.5, 5), model_name=""):
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
        # For each remove fraction:
        for r in remove:
            for exclude_high in [True, False]:
                # for bcq_num in range(1, 6):
                evals.append({'fraction': r,
                              # 'bcq_num': bcq_num,
                              'exclude_high': exclude_high,
                              'avg_reward': self.remove_fraction_eval(dataset,
                                                                      data_values,
                                                                      exclude_high,
                                                                      remove=r)})
                # bcq_num=bcq_num)})
                pd.DataFrame(evals).to_csv(file_path)

    def data_valuate(self, dataset, batch_size):
        """
        Estimate the value of each sample in the specified data set
        """
        print('save data values')
        file_path = '%s/dvrl_%s_train_%d.json' % (self.results_dir, self.dict["env"], len(dataset))
        data_values = []

        for i in tqdm(range(dataset.size)):


            s, a, s_, r, nd = dataset.sample(batch_size, ind=i, to_device=False)#O

            with torch.no_grad():
                batch_values = self.dve_model(concat(s, s_, a, r, nd, self.device))
            data_values.append(detach(batch_values))
        data_values = np.array(data_values)
        dvrl_out = [str(data_values[i]) for i in range(data_values.shape[0])]
        json.dump(dvrl_out, open(file_path, 'w'), indent=4)
        return data_values
