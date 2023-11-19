""" Data Valuation based Batch-Constrained Reinforcement Learning """
from data_valuation.utils.utils import detach, concat, concat_marginal_information
from data_valuation.agents.bcq_dataloader import BCQ
from data_valuation.agents.bear import BEAR, GaussianPolicy
from data_valuation.utils import utils
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json
# from utils.early_stopping import EarlyStopping

import copy
import gym
import os


#from DVBCQ_AAMAS.data_valuation.utils.replay_buffer import ReplayBuffer
#from DVBCQ_AAMAS.data_valuation.agents.reinforce import REINFORCE

from data_valuation.utils.replay_buffer_dataLoader import ReplayBuffer
from data_valuation.agents.reinforce import REINFORCE
from data_valuation.predictor import Predictor
from data_valuation.utils.utils import worker_init_fn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


def calc_moving_average(series, ws_size):
    return series.rolling(ws_size, min_periods=1).mean()


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
        self.env = env

        # self.predictor_scaler = torch.cuda.amp.GradScaler()
        # self.dve_scaler = torch.cuda.amp.GradScaler()

        # Used to store samples selected by the value estimator in each 
        # outer iteration.
        self.sampling_replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, None, device)


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
        # self.dve_optimizer = optim.AdamW(self.dve_model.parameters(), self.dict['dve_lr'])
        self.predictor_optimizer = optim.Adam(self.predictor.parameters(), self.dict['predictor_lr'])
        # self.predictor_optimizer = optim.AdamW(self.predictor.parameters(), self.dict['predictor_lr'])

        if self.dict['marginal_info'] and self.dict['trained_predictor_path'] != None:
            self.load_predictor()
        elif self.dict['marginal_info']: 
            self.train_predictor()

        # Get batch constrained reinforcement learning model 
        # (currently BCQ is the only option, more to be added)
        if self.dict['rl_model'] == 'BCQ':
            self.rl_model = self.get_bcq()

        elif self.dict['rl_model'] == 'BEAR':
            self.rl_model = self.get_bear()
        else:
            raise ValueError("The given rl_method is not supported!")

        self.final_buff =  self.get_final_transitions(source_dataset)

    def get_final_transitions(self, dataset):
        rbuf = ReplayBuffer(self.env.observation_space.shape[0], self.env.action_space.shape[0], None, self.device)
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
                   self.dict['phi'])

    def get_bear(self):
        """
        returns new bcq instance
        """
        return BEAR(self.state_dim,
                   self.action_dim,
                   self.action_space,
                   self.device,
                   self.dict['discount'],
                   self.dict['tau'],
                   self.dict)
                   # self.dict['phi'])

    def train_predictor(self):
        BCELoss = torch.nn.BCELoss()
        dataloader = DataLoader(self.target_dataset, batch_size=self.dict['predictor_batch_size'], shuffle=False)
        # it = iter(dataloader)
        # states, actions, next_states, rewards, terminals = next(it)
        for i in tqdm(range(self.dict['predictor_train_iterations'])):

                it = iter(dataloader)
                states, actions, next_states, rewards, terminals = next(it)

            # for states, actions, next_states, rewards, terminals in dataloader:
            # states, actions, next_states, rewards, terminals = self.valid_dataset.sample(self.dict['predictor_batch_size'])
            # next_states_, rewards_, actions_, terminals_ = self.predictor(states)
                next_states_, rewards_, terminals_ = self.predictor(torch.cat((states, actions), 1))
                self.predictor_optimizer.zero_grad()
                loss_next_state = F.smooth_l1_loss(next_states_, next_states)
                # loss_actions = F.smooth_l1_loss(actions_, actions)
                loss_rewards = F.smooth_l1_loss(rewards_, rewards)
                loss_terminals = BCELoss(terminals_, terminals)
                # loss = loss_next_state + loss_actions + loss_rewards + loss_terminals
                loss = loss_next_state + loss_rewards + loss_terminals
                loss.backward()
                # self.predictor_scaler.scale(loss).backward()
                self.predictor_optimizer.step()
                # self.predictor_scaler.step(self.predictor_optimizer)
                # self.predictor_scaler.update()

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
        est_data_value = self.dve_model.forward(x)
        self.dve_optimizer.zero_grad()

        prob = torch.sum(s_input * torch.log(est_data_value + self.dict['epsilon']) +\
            (1 - s_input) * torch.log(1 - est_data_value + self.dict['epsilon']))

        loss = (-reward * prob) + \
            1e3 * (utils.get_maximum(torch.mean(est_data_value)-self.dict['threshold'], 0) +
                   utils.get_maximum(1-torch.mean(est_data_value)-self.dict['threshold'], 0))

        loss.backward()
        # self.dve_scaler.scale(loss).backward()
        self.dve_optimizer.step()
        # self.dve_scaler.step(self.dve_optimizer)
        # self.dve_scaler.update()

    def train_baseline(self):
        training_iters = 0
        results = []
        bl_path = self.model_dir + "baseline_model/"
        Path(bl_path).mkdir(parents=True, exist_ok=True)
        values = []
        keys = []
        dataloader = DataLoader(self.final_buff, batch_size=self.final_buff.size, shuffle=False, num_workers=0)
        it = iter(dataloader)
        states, actions, next_states, rewards, terminals = next(it)
        # states, _, _, rewards, _ = self.final_buff.sample(self.final_buff.size)
        for r in rewards:
            keys.append(str(detach(r)[0]))
        
        while training_iters < self.dict['max_timesteps']:
            if self.dict['rl_model'] == 'BCQ':
                actor_loss, critic_loss, vae_loss = self.bl_model.train(
                                           replay_buffer=self.source_dataset,
                                           iterations=int(self.dict['eval_freq']),
                                           batch_size=self.dict['mini_batch_size'],
                                           disable_tqdm=False)
            elif self.dict['rl_model'] == 'BEAR':
                vae_loss = None

                prior = GaussianPolicy(self.state_dim,
                                       self.action_dim,
                                       self.dict["hidden_size"],
                                       self.action_space).to(self.dict["device"])

                actor_loss, critic_loss, dual_lambda, mm_dist = self.bl_model.train(
                                           replay_buffer=self.source_dataset,
                                           iterations=int(self.dict['eval_freq']),
                                           batch_size=self.dict['mini_batch_size'],
                                           prior=prior,
                                           m=self.dict['m'],
                                           n=self.dict['n'],
                                           disable_tqdm=False)
            else:
                raise ValueError("The given rl_method is not supported!")

            r = self.eval_policy(self.bl_model)
            training_iters += int(self.dict['eval_freq'])


            results.append({'iteration':training_iters,
                            'reward':r,
                            'actor_loss':actor_loss,
                            'critic_loss':critic_loss,
                            'vae_loss':vae_loss
                            })
            print(f"Training iterations: {training_iters},\
                    Avg Reward: {r},\
                    Actor Loss: {actor_loss}\
                    Critic Loss: {critic_loss}\
                    VAE Loss: {vae_loss}")
            pd.DataFrame(results).to_csv(bl_path+"bl_results.cvs")
            estimates = {}
            # print(self.final_buff.size)
            #print(rewards)
            value_estimates = self.bl_model.get_value_estimate(states)
            #print(value_estimates)
            for key, v  in zip(keys, value_estimates):
                estimates.update({key:v[0]})
            values.append(estimates)
            pd.DataFrame(values).to_csv(bl_path+"values.cvs")
        self.bl_model.save(bl_path)
        print(f"Basline Model Path: {bl_path}")

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
        evaluations = []

        probs = []
        est_dv_curr_values = []
        # indexes = []
        
        # Calculate a(n initial) baseline
        if self.dict['rl_model'] == 'BCQ':
            self.bl_model = self.get_bcq()
        elif self.dict['rl_model'] == 'BEAR':
            self.bl_model = self.get_bear()
        else:
            raise ValueError("The given rl_method is not supported!")


        if self.dict['trained_baseline_path'] != None:
            print('loading baseline model.')
            self.bl_model.load(self.dict['trained_baseline_path'])
        else:
            print('start training baseline model.')
            self.train_baseline()

        if self.dict['rl_model'] == 'BCQ':
            bl_actor_loss, bl_critic_loss, bl_vae_loss = self.bl_model.train(self.target_dataset,
                                                                             int(self.source_dataset_size / self.dict['mini_batch_size']),
                                                                             batch_size=self.dict['mini_batch_size'],
                                                                             optimize=False) # Optimize set to false, 
                                                                                            # since we only want these 
                                                                                            # values from computing the reinforce reward.
                                                                                            # If set to true, then you are training or 
                                                                                            # model on the valid_dataset, + without 
                                                                                            # excluding samples!!!
            baseline_loss = bl_critic_loss
            print("Evaluating BCQ Baseline")
            bcq_eval = self.eval_policy(self.bl_model)

        elif self.dict['rl_model'] == 'BEAR':
            prior = GaussianPolicy(self.state_dim, self.action_dim,
                                   self.dict["hidden_size"], self.action_space).to(self.dict["device"])
            bl_actor_loss, bl_critic_loss, dual_lambda, mmd = self.bl_model.train(self.target_dataset,
                                                                                  int(self.target_dataset.size /
                                                                                      self.dict[
                                                                                                    'mini_batch_size']),
                                                                                  batch_size=self.dict[
                                                                                                'mini_batch_size'],
                                                                                  prior=prior,
                                                                                  optimize=False)
        else:
            raise ValueError("The given rl_method is not supported!")

        state, action, next_state, reward, not_done = self.source_dataset

        state, action, next_state, reward, not_done = torch.FloatTensor(state), \
                                                       torch.FloatTensor(action),\
                                                       torch.FloatTensor(next_state),\
                                                       torch.FloatTensor(reward),\
                                                       torch.FloatTensor(not_done)

        my_dataset = TensorDataset(state, action, next_state, reward, not_done)  # create your datset
        dataloader = DataLoader(my_dataset, batch_size=self.dict['batch_size'], shuffle=False, num_workers=0)
        generator = iter(dataloader)

        # start training
        for iteration in tqdm(range(self.dict['outer_iterations'])):
        # for iteration, j in enumerate(tqdm(data_loader)):
            # batch select
            # states, actions, next_states, rewards, terminals = self.source_dataset.sample(self.dict['batch_size'], to_device=False)




            try:
                # Samples the batch
                states, actions, next_states, rewards, terminals = next(generator)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(my_dataset)
                states, actions, next_states, rewards, terminals = next(generator)


            if self.dict['noise_batch_size'] > 0:
                states, actions, noisy_samples_indexes = utils.add_noise(states,
                                                                         actions,
                                                                         self.dict['state_nr'],
                                                                         self.dict['action_nr'],
                                                                         self.dict['noise_batch_size'],
                                                                         noise_case=self.dict['noise_case'])



            if self.dict['marginal_info']:
                # next_states_pred, rewards_pred, actions_pred, terminals_pred = self.predictor(self.to_device(states))
                next_states_pred, rewards_pred, terminals_pred = self.predictor(
                    self.to_device(np.concatenate([states, actions], 1)))

                dvrl_input = concat_marginal_information(states, 
                                                         next_states, 
                                                         actions, 
                                                         rewards, 
                                                         terminals, 
                                                         detach(next_states_pred), 
                                                         detach(rewards_pred), 
                                                         # detach(actions_pred),
                                                         detach(terminals_pred),
                                                         self.device)
            else:
                # dvrl_input = concat(states, next_states, actions, rewards, terminals, self.device)
                dvrl_input = torch.cat((states, next_states, actions, rewards, terminals), 1).to(self.device)

            est_dv_curr = self.dve_model.forward(dvrl_input)

            # Samples the selection probability
            sel_prob_curr = np.random.binomial(1, detach(est_dv_curr), est_dv_curr.shape)

            # Exception (When selection probability is 0)
            if np.sum(sel_prob_curr) == 0:
                est_dv_curr = 0.5 * np.ones(np.shape(detach(est_dv_curr)))
                sel_prob_curr = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)


            est_dv_curr_values = np.append(est_dv_curr_values, detach(torch.as_tensor(est_dv_curr)).flatten())
            probs = np.append(probs, sel_prob_curr.flatten())

            print(detach(torch.as_tensor(est_dv_curr)).flatten())
            print(sel_prob_curr.flatten())

            # Reset (empty) the sampling buffer, and then add a samples based on
            # the selection probabilities. 
            #print(detach(est_dv_curr))
            #print(sel_prob_curr)
            self.reset_sampling_buffer(detach(states), detach(actions), detach(rewards), detach(next_states), detach(terminals), sel_prob_curr)
            
            # train rl_model with sampled batch data for inner_iterations
            #print("Training Selective Model")
            if self.dict['rl_model'] == 'BCQ':
                _ = self.rl_model.train(replay_buffer=self.sampling_replay_buffer.get_all(),
                                        iterations=self.dict['inner_iterations'],
                                        batch_size=min(self.dict['mini_batch_size'],
                                         self.sampling_replay_buffer.size))
            
                # Compute loss on validation dasaset, but don't optimize
                actor_loss, critic_loss, vae_loss = self.rl_model.train(self.target_dataset,
                                                                        self.dict['eval_loss_iterations'],
                                                                        batch_size=self.dict['mini_batch_size'],
                                                                        optimize=False)


            elif self.dict['rl_model'] == 'BEAR':
                prior = GaussianPolicy(self.state_dim, self.action_dim,
                                       self.dict["hidden_size"], self.action_space).to(self.dict["device"])
                _ = self.rl_model.train(replay_buffer=self.sampling_replay_buffer,
                                        iterations=self.dict['inner_iterations'],
                                        batch_size=min(self.dict['mini_batch_size'], self.sampling_replay_buffer.size),
                                        prior=prior,
                                        m=self.dict['m'],
                                        n=self.dict['n'])

                # Compute loss on validation dasaset, but don't optimize
                actor_loss, critic_loss, dual_lambda, mmd_dist = self.rl_model.train(replay_buffer=self.target_dataset,
                                                                                     iterations=self.dict['eval_loss_iterations'],
                                                                                     batch_size=self.dict['mini_batch_size'],
                                                                                     prior=prior,
                                                                                     m=self.dict['m'],
                                                                                     n=self.dict['n'],
                                                                                     optimize=False)


            mean_loss = critic_loss

            # Compute reward for the Reinforce agent. T
            # The lower the mean_loss is compared to the baseline, the larger the reward.
            # num_selected = len(sel_prob_curr[sel_prob_curr>0])

            # The lower the mean_loss, the larger the reward.
            reinforce_reward = baseline_loss - mean_loss

            # Evaluate the updated policy and save evaluations

            if iteration % self.dict['print_eval'] == 0:
                print(f"Baseline: {baseline_loss}, Mean Loss: {mean_loss}, Reward: {reinforce_reward}")
                print("Evaluating DVBCQ")
                dvbcq_eval = self.eval_policy(self.rl_model)
            else:
                dvbcq_eval = bcq_eval = None

            evals =  {'bcrl_eval_reward':dvbcq_eval,
                     'bcq_eval_reward':bcq_eval,
                     'baseline_loss':baseline_loss,
                     'mean_loss':mean_loss,
                     'reinforce_reward':reinforce_reward}
                     
            evaluations.append(evals)

            pd.DataFrame(evaluations).to_csv(f"{self.results_dir}/evaluations.csv")


            pd.DataFrame(est_dv_curr_values).to_csv(f"{self.results_dir}/est_dv_curr_values.csv")
            pd.DataFrame(probs).to_csv(f"{self.results_dir}/probs.csv")
            # pd.DataFrame(indexes).to_csv(f"{self.results_dir}/noisy_indexes.csv")

            # baseline = EEE

            # If a rolling average baseline is being used, then update the rolling avg.
            # if self.dict['baseline'] == 'rolling_avg':
            #     baseline = (1.0-self.dict['tau_baseline']) * baseline + self.dict['tau_baseline'] * mean_loss
            #     reinforce_reward = reinforce_reward - baseline

            # Optimize the data value estimator
            self.train_dve(dvrl_input, self.to_device(sel_prob_curr), reinforce_reward)


        # Save the RL model
        self.rl_model.save(self.model_dir) # Save batch constrained RL model
        self.save_dve() # Save data value estimator

    def reset_sampling_buffer(self, states, actions, rewards, next_states, terminals, sel_prob_curr):
        self.sampling_replay_buffer.reset()
        for s, a, r, s_, t, sp in zip(states, actions, rewards, next_states, terminals, sel_prob_curr):
            if sp > .0: # If selected
                self.sampling_replay_buffer.add(s, a, s_, r, 1-t)          

    def save_dve(self):
        """
        Save reinforce model
        """
        torch.save(self.dve_model.state_dict(), self.model_dir + "_reinforce")
        torch.save(self.dve_optimizer.state_dict(), self.model_dir + "_reinforce_optimizer")

    def eval_policy(self, model, eval_episodes=10):
        """
        Runs policy for X episodes and returns average reward
        A fixed seed is used for the eval environment
        """
        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = self.env.reset(), False
            while not done:
                action = model.select_action(np.array(state))
                state, reward, done, _ = self.env.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def remove_fraction_eval(self, dataset, data_values, reverse=False, remove=0.1):
        """
        Evaluate model trained with a fraction of samples excluded.
        reverse: False results in excluding lowest samples, true excluded highest
        """
        print(f"Constructing Dataset - Excluding High Value {reverse} - Proportion {remove}")
        dataset_tmp = []

        print(data_values)
        print(data_values.shape)

        # for i in range(dataset.size):
        for i in range(len(data_values)):
            # s, a, s_, r, nd = dataset.sample(1, ind=i, to_device=False)
            s, a, s_, r, nd = dataset.__getitem__(i)
            dataset_tmp.append({'s':s, 'a':a, 's_':s_, 'r':r, 'nd':nd, 'v':data_values[i]})
        dataset_tmp = sorted(dataset_tmp, key=lambda k: k['v'], reverse=reverse) 

        # Train batch constrained rl model with a dataset where a specified fraction of 
        # high/low value samples have been removed
        start_idx = int(len(dataset_tmp)*remove)

        dataset_train = ReplayBuffer(self.state_dim, self.action_dim, None, self.device)
        for t in dataset_tmp[start_idx:]:
            dataset_train.add(t['s'], t['a'], t['s_'], t['r'], t['nd'])          

        # Load eval model
        if self.dict['rl_model'] == 'BCQ':
            eval_model = self.get_bcq()
        elif self.dict['rl_model'] == 'BEAR':
            eval_model = self.get_bear()

        print("Training Eval Model")
        _ = eval_model.train(dataset_train.get_all(),
                             self.dict['eval_train_iterations'],
                             batch_size=self.dict['mini_batch_size'])

        # Eval on target domain
        samples_removed = 'low value' if reverse == False else 'high value'
        print("Evaluating %s with %f of %s samples removed" % (self.dict['rl_model'], remove, samples_removed)) 
        self.env.seed(self.dict['seed'] + 100)
        state = self.env.reset()
        rewards = []
        for _ in tqdm(range(self.dict['eval_iterations'])):
            action = eval_model.select_action(state)
            state, reward, _, _ = self.env.step(action)
            rewards.append(reward)         
  
        # Return avg reward 
        return np.mean(rewards)


    def remove_fraction_evals(self, dataset, data_values, remove=np.linspace(0.1, 0.5, 5)):
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
        for r in remove:
            for exclude_high in [True, False]:
                evals.append({'fraction':r,
                              'exclude_high':exclude_high,
                              'avg_reward':self.remove_fraction_eval(dataset,
                                                                     data_values, 
                                                                     exclude_high, 
                                                                     remove=r)})
                pd.DataFrame(evals).to_csv(file_path)
            
    def data_valuate(self, dataset, batch_size):
        """
        Estimate the value of each sample in the specified data set
        """
        print('save data values')
        file_path = '%s/dvrl_%s_train_%d.json' % (self.results_dir, self.dict["env"], len(dataset))
        data_values = []
        for i in tqdm(range(0, dataset.size, batch_size)):
            # s, a, s_, r, nd = dataset.sample(batch_size, ind=i, to_device=False)
            s, a, s_, r, nd = dataset.__getitem__(i)

            if self.dict['marginal_info']:
                next_states_pred, rewards_pred, terminals_pred = self.predictor(self.to_device(np.hstack((s, a))))
                input = self.to_device(np.hstack((s,s_,a,r,nd,
                                                    detach(next_states_pred),
                                                    detach(rewards_pred),
                                                    # detach(actions_pred),
                                                    detach(terminals_pred))))
            else:
                input = self.to_device(np.hstack((s, s_, a, r, nd, self.device)))


            with torch.no_grad():
                # batch_values = self.dve_model.forward(concat(s, s_, a, r, nd, self.device))
                batch_values = self.dve_model.forward(input)
            data_values.append(detach(batch_values))
        data_values = np.array(data_values)
        dvrl_out = [str(data_values[i]) for i in range(data_values.shape[0])]
        json.dump(dvrl_out, open(file_path, 'w'), indent=4)
        return data_values
