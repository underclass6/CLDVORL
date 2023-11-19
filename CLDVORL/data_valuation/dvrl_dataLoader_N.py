""" Data Valuation based Batch-Constrained Reinforcement Learning """
from itertools import cycle

from torch.utils.data import DataLoader, TensorDataset
from data_valuation.utils.utils import detach, concat, concat_marginal_information
from data_valuation.agents.bcq_dataloader import BCQ
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.optim.lr_scheduler as lr_scheduler
from numpy.random import default_rng
from numpy.random import Generator, PCG64
import random

#from DVBCQ_AAMAS.data_valuation.utils.replay_buffer import ReplayBuffer
#from DVBCQ_AAMAS.data_valuation.agents.reinforce import REINFORCE

from data_valuation.utils.replay_buffer_dataLoader import ReplayBuffer
from data_valuation.agents.reinforce import REINFORCE
from data_valuation.predictor import Predictor



def calc_moving_average(series, ws_size):
    return series.rolling(ws_size, min_periods=1).mean()

def get_every_n(a, n=200):
    for i in range(a.shape[0] // n):
        yield a[n*i:n*(i+1)]

class DVRL(object):
    def __init__(self, source_dataset, target_dataset, device, env, results_dir, args={}):
       
        self.dict = vars(args)
        self.args = args
        assert len(self.dict) > 0
        
        # data
        self.model_dir = utils.make_dvrl_dir(results_dir, args)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_space = env.action_space
        self.source_dataset = source_dataset
        self.source_dataset_size = len(source_dataset[0])

        self.target_dataset = target_dataset
        self.target_dataset_size = len(target_dataset[0])

        self.results_dir = results_dir
        self.device = device
        self.target_env = env

        self.shuffle = self.dict["shuffle"]
        self.num_workers = self.dict["num_workers"]



        self.source_env = utils.get_gym(self.dict["env"], self.dict["source_env_friction"], self.dict["target_env_mass_torso"])


        # Used to store samples selected by the value estimator in each outer iteration.
        self.sampling_replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, device)

        # models
        if self.dict['marginal_info']:
            # Since input is a concat of (s, s', a, r, t, s'_predict-s, a-a_predict, r-r_predicts, t-t_predict)
            reinforce_state_dim = self.state_dim * 3 + self.action_dim + 4
        else:
            # Input is a concat of (s, s', a, r, t)
            reinforce_state_dim = self.state_dim * 2 + self.action_dim + 2
        reinforce_action_dim = 1 # Outputs probability for each sample
        self.dve_model = REINFORCE(reinforce_state_dim,
                                   reinforce_action_dim,
                                   layers=self.dict['reinforce_layers'],
                                   args=self.dict).to(device)

        self.predictor = Predictor(self.state_dim,
                                   self.action_dim,
                                   self.action_space,
                                   layers=self.dict['predictor_layers']).to(device)

        # optimizers
        self.dve_optimizer = optim.Adam(self.dve_model.parameters(), self.dict['dve_lr'])
        self.predictor_optimizer = optim.Adam(self.predictor.parameters(), self.dict['predictor_lr'])

        if self.dict['marginal_info'] and self.dict['trained_predictor_path'] != None:
            self.load_predictor()
        elif self.dict['marginal_info']: 
            self.train_predictor()



        self.final_buff_source =  self.get_final_transitions(source_dataset)
        self.final_buff_target =  self.get_final_transitions(target_dataset)

    def get_final_transitions(self, dataset):
        rbuf = ReplayBuffer(self.target_env.observation_space.shape[0], self.target_env.action_space.shape[0], self.device)

        # states, actions, next_states, rewards, terminals = dataset.sample(len(dataset), to_device=False)
        states, actions, next_states, rewards, terminals = dataset
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
                   self.dict['phi'],
                   self.dict['num_workers'],
                   self.dict['shuffle'])


    def train_predictor(self):
        BCELoss = torch.nn.BCELoss()
        for i in tqdm(range(self.dict['predictor_train_iterations'])):
            states, actions, next_states, rewards, terminals = self.target_dataset.sample(self.dict['predictor_batch_size'])
            next_states_, rewards_, terminals_ = self.predictor(torch.cat((states, actions), 1).to(self.device))
            self.predictor_optimizer.zero_grad()
            loss_next_state = F.smooth_l1_loss(next_states_, next_states)
            loss_rewards = F.smooth_l1_loss(rewards_, rewards)
            loss_terminals = BCELoss(terminals_, terminals)
            loss = loss_next_state + loss_rewards + loss_terminals
            loss.backward()
            self.predictor_optimizer.step()

            if i%1000 == 0:
                print(f"Predictor Loss: {loss:.5f}")
                print(f"Next State Loss: {loss_next_state:.10f} | Reward Loss: {loss_rewards:.10f} | Terminal Loss: {loss_terminals:.10f}")
                print(f"Next State: {next_states[0]} Predicted: {next_states_[0]}")
                print(f"Reward: {rewards[0]} Predicted: {rewards_[0]}")
                # print(f"Action: {actions[0]} Predicted: {actions_[0]}")
                print(f"Terminals: {terminals[0]} Predicted: {terminals_[0]}")
        torch.save(self.predictor.state_dict(), self.model_dir + "_predictor")
        print(f"Predictor saved to: {self.model_dir}_predictor")

    def load_predictor(self):
        self.predictor.load_state_dict(torch.load(self.dict['trained_predictor_path'], map_location=torch.device(self.device)))

    def train_dve(self, x, s_input, reward):
        """
         Training data value estimator
         s_input: selection probability
         x_input: Sample tuple
         reward_input: reward signal for training the reinforce agent
        """
        est_data_value = self.dve_model(x)
        self.dve_optimizer.zero_grad()

        prob = torch.sum(s_input * torch.log(est_data_value + self.dict['epsilon']) +\
            (1 - s_input) * torch.log(1 - est_data_value + self.dict['epsilon']))


        loss = (-reward * prob) + 1e3 * (utils.get_maximum(torch.mean(est_data_value)-self.dict['threshold'], 0) +
                   utils.get_maximum(1-torch.mean(est_data_value)-self.dict['threshold'], 0))

        loss.backward()
        self.dve_optimizer.step()

    def train_baseline(self, data="train"):

        bl_training_iter = np.array([])
        bl_avg_reward = np.array([])
        bl_actor_loss = np.array([])
        bl_critic_loss = np.array([])
        bl_vae_loss = np.array([])


        training_iters = 0
        results = []
        bl_path = f"{self.model_dir}basline_model/{data}/"
        Path(bl_path).mkdir(parents=True, exist_ok=True)
        values = []
        keys = []
        if data == "train":
            states, _, _, rewards, _ = self.final_buff_source.sample(self.final_buff_source.size)
            model = self.bl_train_model
            buffer = self.source_dataset
        elif data == "valid":
            states, _, _, rewards, _ = self.final_buff_target.sample(self.final_buff_target.size)
            model = self.bl_valid_model
            buffer = self.target_dataset

        for r in rewards:
            keys.append(str(detach(r)[0]))




        while training_iters < self.dict['max_timesteps']:
            if self.dict['rl_model'] == 'BCQ':
                actor_loss, critic_loss, vae_loss = model.train(
                                           replay_buffer=buffer,
                                           iterations=int(self.dict['eval_freq']),
                                           batch_size=self.dict['mini_batch_size'],
                                           disable_tqdm=False)


            # r = self.eval_policy(model)
            r, _ = self.normalized_eval_policy(model)
            # r = self.normalized_eval_policy_perStep(model)

            training_iters += int(self.dict['eval_freq'])

            bl_training_iter = np.append(bl_training_iter, training_iters)
            bl_avg_reward = np.append(bl_avg_reward, r)
            bl_actor_loss = np.append(bl_actor_loss, actor_loss)
            bl_critic_loss = np.append(bl_critic_loss, critic_loss)
            bl_vae_loss = np.append(bl_vae_loss, vae_loss)

            print(f"Training iterations: {training_iters},\
                    Avg Reward: {r},\
                    Actor Loss: {actor_loss}\
                    Critic Loss: {critic_loss}\
                    VAE Loss: {vae_loss}")

            np.save(f'{self.results_dir}/bl_training_iter', bl_training_iter, allow_pickle=True)
            np.save(f'{self.results_dir}/bl_avg_reward', bl_avg_reward, allow_pickle=True)
            np.save(f'{self.results_dir}/bl_actor_loss', bl_actor_loss, allow_pickle=True)
            np.save(f'{self.results_dir}/bl_critic_loss', bl_critic_loss, allow_pickle=True)
            np.save(f'{self.results_dir}/bl_vae_loss', bl_vae_loss, allow_pickle=True)

            estimates = {}

            value_estimates = model.get_value_estimate(states)

            for key, v  in zip(keys, value_estimates):
                estimates.update({key:v[0]})
            values.append(estimates)

            np.save(f'{self.results_dir}/bl_values',  np.array(values), allow_pickle=True)

        model.save(bl_path)
        print(f"{data}-Basline Model Path: {bl_path}")

    def to_device(self, x):
        """
        Copy x to GPU
        """
        return torch.FloatTensor(x).to(self.device)

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

        bcq_actor_loss = np.array([])
        bcq_critic_loss = np.array([])
        bcq_vae_loss = np.array([])

        dvbcq_actor_loss = np.array([])
        dvbcq_critic_loss = np.array([])
        dvbcq_vae_loss = np.array([])


        rew_deque = deque(maxlen=20)

        update_flag = False
        best_reward_sig = 0.

        reward_signal_history = np.array([0.0])
        baseline_history = np.array([0.0])
        rl_model_rew_history = np.array([])
        # rng = default_rng()
        rng = Generator(PCG64(12345))

        
        # Calculate a(n initial) baseline
        if self.dict['rl_model'] == 'BCQ':
            self.bl_train_model = self.get_bcq()
            self.bl_valid_model = self.get_bcq()


        if self.dict['trained_baseline_path'] != None:
            print('loading baseline model.')
            self.bl_train_model.load(self.dict['trained_baseline_path'])
        else:
            print('start training baseline model.')
            self.train_baseline(data="train")


        if self.dict['trained_valid_baseline_path'] != None:
            print('loading valid baseline model.')
            self.bl_valid_model.load(self.dict['trained_valid_baseline_path'])
        else:
            print('start training valid baseline model.')
            self.train_baseline("valid")


        # if self.dict['rl_model'] == 'BCQ':
        #     bl_actor_loss, bl_critic_loss, bl_vae_loss = self.bl_model.train(self.target_dataset,
        #                                                                      int(self.target_dataset.size / self.dict[
        #                                                                          'mini_batch_size']),
        #                                                                      batch_size=self.dict['mini_batch_size'],
        #                                                                      optimize=False)  # Optimize set to false,
        #     # since we only want these
        #     # values from computing the reinforce reward.
        #     # If set to true, then you are training or
        #     # model on the target_dataset, + without
        #     # excluding samples!!!
        # baseline_loss = bl_critic_loss
        # print("Evaluating BCQ Baseline")
        # bcq_eval = self.eval_policy(self.bl_model)

        # bcq_avg_rew = self.eval_policy(self.bl_train_model, seed_num=10)
        # bcq_avg_rew = self.normalized_eval_policy(self.bl_train_model, seed_num=50)


        bcq_avg_rew, _ = self.normalized_eval_policy(self.bl_valid_model, seed_num=55)
        # bcq_avg_rew = self.normalized_eval_policy_perStep(self.bl_train_model, seed_num=55)


        # bcq_avg_rew = self.eval_policy_perStep(self.bl_model)

        # Get batch constrained reinforcement learning model
        # (currently BCQ is the only option, more to be added)
        # if self.dict['rl_model'] == 'BCQ' and self.dict['trained_baseline_path'] != None:
        if self.dict['rl_model'] == 'BCQ':
            self.rl_model = self.get_bcq()
            # self.rl_model =  copy.deepcopy(self.bl_train_model)
            # self.bk_model = copy.deepcopy(self.bl_model)

        # else:
        #     raise ValueError("The given rl_method is not supported!")
        curr_val_data = {}
        mean_val = 0.0
        global_min = 0.0
        global_max = 1.0
        first=True
        fitted=False
        scaler = StandardScaler()

        flag_update = False
        best_reward = 0.0


        state, action, next_state, reward, not_done = self.source_dataset

        state, action, next_state, reward, not_done = torch.FloatTensor(state), \
                                                       torch.FloatTensor(action),\
                                                       torch.FloatTensor(next_state),\
                                                       torch.FloatTensor(reward),\
                                                       torch.FloatTensor(not_done)

        my_dataset = TensorDataset(state, action, next_state, reward, not_done)  # create your datset
        dataloader = DataLoader(my_dataset, batch_size=self.dict['batch_size'], shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=True)
        data_loader = cycle(dataloader)
        generator = iter(data_loader)


        # start training
        for iteration in tqdm(range(self.dict['outer_iterations'])):
            seed_num = random.randint(0,1000)
            t_start = time.time()
            # batch select
            # states, actions, next_states, rewards, terminals = self.source_dataset.sample(self.dict['batch_size'], seed=iteration, to_device=False)

            states, actions, next_states, rewards, terminals = next(generator)

            states, actions, next_states, rewards, terminals = states.to(self.device), \
                                                          actions.to(self.device), \
                                                          next_states.to(self.device), \
                                                          rewards.to(self.device), \
                                                          terminals.to(self.device)

            # dvrl_input = concat(states, next_states, actions, rewards, terminals, self.device)
            dvrl_input = torch.cat((states, next_states, actions, rewards, terminals), 1).to(self.device)

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

            # self.reset_sampling_buffer(states, actions, rewards, next_states, terminals, sel_pr``ob_curr)
            # if self.dict["sampler_dist"] == "binomial":
            #     self.reset_sampling_buffer(states, actions, rewards, next_states, terminals, sel_prob_curr.flatten())
            # elif self.dict["sampler_dist"] == "bernoulli":
            #     self.reset_sampling_buffer(states, actions, rewards, next_states, terminals, detach(sel_prob_curr).flatten())

            self.reset_sampling_buffer(detach(states), detach(actions), detach(rewards), detach(next_states),  detach(terminals), sel_prob_curr.flatten())

            # train rl_model with sampled batch data for inner_iterations
            #print("Training Selective Model")
            if self.dict['rl_model'] == 'BCQ':

                # bl_actor_loss, bl_critic_loss, bl_vae_loss = self.bl_model.train(self.target_dataset,
                # bl_actor_loss, bl_critic_loss, bl_vae_loss = self.bl_model.train(curr_val_data,
                #                                                         self.dict['eval_loss_iterations'],
                #                                                         batch_size=self.dict['mini_batch_size'],
                #                                                         optimize=False)

                # bcq_avg_rew = self.normalized_eval_policy(self.bl_valid_model, "BCQ", eval_episodes=10, seed_num=seed_num)

                _ = self.rl_model.train(replay_buffer=self.sampling_replay_buffer.get_all(),
                                        iterations=self.dict['inner_iterations'],
                                        batch_size=min(self.dict['mini_batch_size'],
                                         self.sampling_replay_buffer.size))


                # Compute loss on validation dasaset, but don't optimize
                # actor_loss, critic_loss, vae_loss = self.rl_model.train(self.target_dataset,
                # actor_loss, critic_loss, vae_loss = self.rl_model.train(curr_val_data,
                #                                                         self.dict['eval_loss_iterations'],
                #                                                         batch_size=self.dict['mini_batch_size'],
                #                                                         optimize=False)

                # dvbcq_avg_rew = self.eval_policy(self.rl_model, "DVBCQ", eval_episodes=1, seed_num=seed_num)

                dvbcq_avg_rew, dvbcq_actual_rew = self.normalized_eval_policy(self.rl_model, "DVBCQ", eval_episodes=3, seed_num=seed_num)
                # dvbcq_avg_rew = self.normalized_eval_policy_perStep(self.rl_model, "DVBCQ", eval_episodes=3, seed_num=seed_num)





            # bcq_actor_loss = np.append(bcq_actor_loss, bl_actor_loss)
            # bcq_critic_loss = np.append(bcq_critic_loss, bl_critic_loss)
            # bcq_vae_loss = np.append(bcq_vae_loss, bl_vae_loss)
            #
            # dvbcq_actor_loss = np.append(dvbcq_actor_loss, actor_loss)
            # dvbcq_critic_loss = np.append(dvbcq_critic_loss, critic_loss)
            # dvbcq_vae_loss = np.append(dvbcq_vae_loss, vae_loss)


            # Compute reward for the Reinforce agent.
            # The lower the mean_loss, the larger the reward.

            reinforce_reward = dvbcq_avg_rew - bcq_avg_rew
            # reinforce_reward = dvbcq_avg_rew



            # if reinforce_reward > 0:
            reward_signal_history = np.append(reward_signal_history, reinforce_reward)
            rew_deque.append(reinforce_reward)
            rl_model_rew_history = np.append(rl_model_rew_history, dvbcq_avg_rew)
            baseline_history = np.append(baseline_history, bcq_avg_rew)




            # Evaluate the updated policy and save evaluations

            if iteration % self.dict['print_eval'] == 0:
                print(" ")
                # print(f"Baseline: {bl_critic_loss}, Mean Loss: {critic_loss}, Reinforce Reward: {reinforce_reward}")
                print(f"BCQ Reward: {bcq_avg_rew}, DVBCQ Reward: {dvbcq_avg_rew}, Reinforce Reward: {reinforce_reward}")
                print("Evaluating DVBCQ")
                # dvbcq_eval = self.eval_policy(self.rl_model, eval_episodes=10, seed_num=seed_num+2)

                dvbcq_eval, dvbcq_eval_un = self.normalized_eval_policy(self.rl_model, model_name="DVBCQ", eval_episodes=10, seed_num=seed_num+2)
                # dvbcq_eval = self.normalized_eval_policy_perStep(self.rl_model, model_name="DVBCQ", eval_episodes=10, seed_num=seed_num+2)

                # dvbcq_eval = self.eval_policy_perStep(self.rl_model)
                print("Evaluating BCQ")
                # bcq_eval = self.eval_policy(self.bl_train_model, eval_episodes=10, seed_num=seed_num + 2)

                bcq_eval, bcq_eval_un = self.normalized_eval_policy(self.bl_valid_model, model_name="BCQ", eval_episodes=10, seed_num=seed_num + 2)
                # bcq_eval = self.normalized_eval_policy_perStep(self.bl_train_model, model_name="BCQ", eval_episodes=10, seed_num=seed_num + 2)

                # bcq_eval = self.eval_policy_perStep(self.bl_model)
            else:
                bcq_eval_un = dvbcq_eval_un = None

            bcq_evals = np.append(bcq_evals, bcq_eval_un)
            dvbcq_evals = np.append(dvbcq_evals, dvbcq_eval_un)
            bcq_rews = np.append(bcq_rews, bcq_avg_rew)
            dvbcq_rews = np.append(dvbcq_rews, dvbcq_avg_rew)
            reinforce_rewards = np.append(reinforce_rewards, reinforce_reward)

            # baseline_losses = np.append(baseline_losses, bl_critic_loss)
            # if flag_update:
            print("----------------------------------------")
            print(f"Reinforce reward: {reinforce_reward}")
            print("----------------------------------------")
            if self.dict["sampler_dist"] == "binomial":
                self.train_dve(dvrl_input, self.to_device(sel_prob_curr), reinforce_reward)
            # self.train_dve(dvrl_input, sel_prob_curr, reinforce_reward)
            elif self.dict["sampler_dist"] == "bernoulli":
                self.train_dve(dvrl_input, sel_prob_curr, reinforce_reward)

            bcq_avg_rew = (self.dict['T'] - 1) * bcq_avg_rew / self.dict['T'] + dvbcq_avg_rew / self.dict['T']
            # bcq_avg_rew = (1.0 - self.dict['tau']) * bcq_avg_rew + self.dict['tau'] * dvbcq_avg_rew


            # mean_val = np.mean(np.array(rl_model_rew_history).reshape(-1, 1))
            # bcq_avg_rew = mean_val + (1 / (len(rl_model_rew_history) + 1)) * (dvbcq_avg_rew - mean_val)


            t_end = time.time()

            # If a rolling average baseline is being used, then update the rolling avg.
            # if self.dict['baseline'] == 'rolling_avg':
            #     bcq_avg_rew = (self.dict['T'] - 1) * bcq_avg_rew / self.dict['T'] + dvbcq_avg_rew / self.dict['T']

            print(f"Iteration:{iteration} took %.2f seconds" % (t_end - t_start))


            np.save(f'{self.results_dir}/dvbcq_evals', dvbcq_evals, allow_pickle=True)
            np.save(f'{self.results_dir}/bcq_evals', bcq_evals, allow_pickle=True)
            np.save(f'{self.results_dir}/dvbcq_eval_reward', dvbcq_rews, allow_pickle=True)
            np.save(f'{self.results_dir}/bcq_eval_reward', bcq_rews, allow_pickle=True)
            # np.save(f'{self.results_dir}/baseline_loss', baseline_losses, allow_pickle=True)
            # np.save(f'{self.results_dir}/mean_loss', mean_losses, allow_pickle=True)
            np.save(f'{self.results_dir}/reinforce_reward', reinforce_rewards, allow_pickle=True)
            np.save(f'{self.results_dir}/est_dv_curr_values', est_dv_curr_values, allow_pickle=True)
            np.save(f'{self.results_dir}/probs', probs, allow_pickle=True)

            # np.save(f'{self.results_dir}/bcq_actor_loss', bcq_actor_loss, allow_pickle=True)
            # np.save(f'{self.results_dir}/bcq_critic_loss', bcq_critic_loss, allow_pickle=True)
            # np.save(f'{self.results_dir}/bcq_vae_loss', bcq_vae_loss, allow_pickle=True)
            # np.save(f'{self.results_dir}/dvbcq_actor_loss', dvbcq_actor_loss, allow_pickle=True)
            # np.save(f'{self.results_dir}/dvbcq_critic_loss', dvbcq_critic_loss, allow_pickle=True)
            # np.save(f'{self.results_dir}/dvbcq_vae_loss', dvbcq_vae_loss, allow_pickle=True)


        # Save the RL model
        self.rl_model.save(self.model_dir) # Save batch constrained RL model
        self.save_dve() # Save data value estimator



    def reset_sampling_buffer(self, states, actions, rewards, next_states, terminals, sel_prob_curr):
        self.sampling_replay_buffer.reset()
        for s, a, r, s_, t, sp in zip(states, actions, rewards, next_states, terminals, sel_prob_curr):
            # if sp > 0.0: # If selected
            if int(sp): # If selected
                self.sampling_replay_buffer.add(s, a, s_, r, 1-t)

    def save_dve(self):
        """
        Save reinforce model
        """
        torch.save(self.dve_model.state_dict(), self.model_dir + "_reinforce")
        torch.save(self.dve_optimizer.state_dict(), self.model_dir + "_reinforce_optimizer")

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

    def normalized_eval_policy(self, model, model_name="", eval_episodes=10, seed_num=None, disable_tqdm=True):
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
        # norm_avg_rewards = (episodes_rewards - np.min(episodes_rewards)) / (np.max(episodes_rewards) - np.min(episodes_rewards))
        episodes_rewards -= np.mean(episodes_rewards)
        # episodes_rewards /= np.std(episodes_rewards)


        # norm_avg_reward = (episodes_rewards - np.mean(episodes_rewards)) / (np.std(episodes_rewards) + 1e-10)
        norm_avg_reward = np.mean(episodes_rewards)

        print("---------------------------------------")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: Normalized Average Reward {norm_avg_reward:.3f}")
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


        avg_reward_m -= np.mean(avg_reward_m)
        norm_avg_reward = np.mean(avg_reward_m)

        print("---------------------------------------")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: Normalized Average Reward {norm_avg_reward:.3f}")
        print(f"{model_name}: Evaluation over {eval_episodes} episodes: Average Reward {avg_reward:.3f}")
        print("---------------------------------------")
        return norm_avg_reward



    def remove_fraction_eval(self, dataset, data_values, reverse=False, remove=0.1):
        """
        Evaluate model trained with a fraction of samples excluded.
        reverse: False results in excluding lowest samples, true excluded highest
        """
        print(f"Constructing Dataset - Excluding High Value {reverse} - Proportion {remove}")
        dataset_tmp = []

        # print(data_values)
        # print(data_values.shape)

        # for i in range(dataset.size):
        state, action, next_state, reward, not_done = dataset
        source_size = len(state)

        # state, action, next_state, reward, not_done = torch.FloatTensor(state), \
        #                                                torch.FloatTensor(action),\
        #                                                torch.FloatTensor(next_state),\
        #                                                torch.FloatTensor(reward),\
        #                                                torch.FloatTensor(not_done)
        #
        # my_dataset = TensorDataset(state, action, next_state, reward, not_done)  # create your datset
        # dataloader = DataLoader(my_dataset, batch_size=1, shuffle=self.shuffle, num_workers=self.num_workers)
        # data_loader = cycle(dataloader)
        # generator = iter(data_loader)


        for i in range(len(data_values)):
            # s, a, s_, r, nd = next(generator)
            # # s, a, s_, r, nd = dataset.sample(1, ind=i, to_device=False)
            # s, a, s_, r, nd = s.to(self.device), \
            #                    a.to(self.device), \
            #                    s_.to(self.device), \
            #                    r.to(self.device), \
            #                    nd.to(self.device)
            s, a, s_, r, nd = state[i], action[i], next_state[i], reward[i], not_done[i]

            dataset_tmp.append({'s':s, 'a':a, 's_':s_, 'r':r, 'nd':nd, 'v':data_values[i]})
        dataset_tmp = sorted(dataset_tmp, key=lambda k: k['v'], reverse=reverse) 

        # Train batch constrained rl model with a dataset where a specified fraction of 
        # high/low value samples have been removed
        start_idx = int(len(dataset_tmp)*remove)

        dataset_train = ReplayBuffer(self.state_dim, self.action_dim, self.device)
        for t in dataset_tmp[start_idx:]:
            dataset_train.add(t['s'], t['a'], t['s_'], t['r'], t['nd'])          

        # Load eval model
        # if self.dict['rl_model'] == 'BCQ' and self.dict['trained_baseline_path'] != None:
        if self.dict['rl_model'] == 'BCQ':
            # eval_model = copy.deepcopy(self.bl_train_model)
            eval_model = self.get_bcq()
            # eval_model = copy.deepcopy(self.bl_train_model)

        print("Training Eval Model")
        _ = eval_model.train(dataset_train.get_all(),
                             self.dict['eval_train_iterations'],
                             batch_size=self.dict['mini_batch_size'])

        # Eval on target domain
        samples_removed = 'low value' if reverse == False else 'high value'
        print("Evaluating %s with %f of %s samples removed" % (self.dict['rl_model'], remove, samples_removed)) 
        # self.target_env.seed(self.dict['seed'] + 100)
        # state = self.target_env.reset()
        # rewards = []
        # rewards = 0.
        # for _ in tqdm(range(self.dict['eval_iterations'])):
        #     action = eval_model.select_action(state)
        #     state, reward, _, _ = self.target_env.step(action)
        #     # rewards.append(reward)
        #     rewards += reward

        # avg_reward = self.eval_policy(eval_model, "DVBCQ_Plot", eval_episodes=100, seed_num=0)


        avg_reward = self.eval_policy(eval_model, "DVBCQ_Plot", eval_episodes=100, seed_num=225577, disable_tqdm=False, env="target")
        # avg_reward = self.eval_policy(eval_model, f"DVBCQ_Plot for BCQ num: {bcq_num}", eval_episodes=100, seed_num=225577, disable_tqdm=False, env="target")



        # avg_rewards /= self.dict['eval_iterations']
        # rewards avg reward
        # rewards /= self.dict['eval_iterations']


        # return np.mean(rewards)
        return avg_reward


    def remove_fraction_evals(self, dataset, data_values, remove=np.linspace(0.0, 0.5, 6)):
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
        file_path = self.results_dir + '/' + self.dict["env"] + '_evaluations.csv'
        evals = [] 
        # For each remove fraction:

        for r in tqdm(remove):
            print("fraction:", r)
            for exclude_high in tqdm([True, False]):
                print(r, exclude_high)
                # for bcq_num in range(1, 6):
                evals.append({'fraction':r,
                              # 'bcq_num': bcq_num,
                              'exclude_high':exclude_high,
                              'avg_reward':self.remove_fraction_eval(dataset,
                                                                     data_values,
                                                                     exclude_high,
                                                                     remove=r)})
                                                                     # bcq_num=bcq_num)})

                pd.DataFrame(evals).to_csv(file_path)
            
    def data_valuate(self, dataset, batch_size):
        """
        Estimate the value of each sample in the specified data set
        """

        state, action, next_state, reward, not_done = dataset
        source_size = len(state)
        print('save data values')
        file_path = '%s/dvrl_%s_train_%d.json' % (self.results_dir, self.dict["env"],source_size)
        data_values = []




        # state, action, next_state, reward, not_done = torch.FloatTensor(state), \
        #                                                torch.FloatTensor(action),\
        #                                                torch.FloatTensor(next_state),\
        #                                                torch.FloatTensor(reward),\
        #                                                torch.FloatTensor(not_done)

        # my_dataset = TensorDataset(state, action, next_state, reward, not_done)  # create your datset
        # dataloader = DataLoader(my_dataset, batch_size=1, shuffle=self.shuffle, num_workers=self.num_workers)
        # data_loader = cycle(dataloader)
        # generator = iter(data_loader)

        for i in tqdm(range(0, source_size, batch_size)):
            s, a, s_, r, nd = state[i], action[i], next_state[i], reward[i], not_done[i]

            # s, a, s_, r, nd = dataset.sample(batch_size, ind=i, to_device=False)

            # next_states_pred, rewards_pred, terminals_pred = self.predictor(self.to_device(np.hstack((s, a))))

            # if self.dict['marginal_info']:
            #     input = self.to_device(torch.from_numpy(np.hstack((s,s_,a,r,nd,
            #                                         detach(next_states_pred),
            #                                         detach(rewards_pred),
            #                                         # detach(actions_pred),
            #                                         detach(terminals_pred)))))
            # else:
            #     input = (s, s_, a, r, nd)
            # s, a, s_, r, nd = next(generator)

            # s, a, s_, r, nd = s.to(self.device), \
            #                    a.to(self.device), \
            #                    s_.to(self.device), \
            #                    r.to(self.device), \
            #                    nd.to(self.device)

            # s, a, s_, r, nd = dataset.sample(batch_size, ind=i, to_device=False)
            with torch.no_grad():
                batch_values = self.dve_model(concat(s, s_, a, r, nd, self.device))
                # batch_values = self.dve_model(torch.cat((s, s_, a, r, nd), 1).to(self.device))
            data_values.append(detach(batch_values))
        data_values = np.array(data_values)
        dvrl_out = [str(data_values[i]) for i in range(data_values.shape[0])]
        json.dump(dvrl_out, open(file_path, 'w'), indent=4)
        return data_values
