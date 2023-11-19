import copy
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm




class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device, phi=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        # self.max_action_n = max_action
        # self.max_action = max_action
        self.max_action = torch.from_numpy(max_action).to(device)
        self.phi = phi


    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat((state, action), 1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)
        # return torch.clamp(a + action, -self.max_action_n, self.max_action_n)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)


    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat((state, action), 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat((state, action), 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat((state, action), 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        #self.max_action = max_action
        self.max_action = torch.from_numpy(max_action).to(device)
        self.latent_dim = latent_dim
        self.device = device


    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat((state, action), 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        # z = mean + std * torch.randn_like(std)
        z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(self.device)

        u = self.decode(state, z)

        return u, mean, std


    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            # z = torch.randn((state.shape[0], self.latent_dim)).clamp(-0.5,0.5).to(self.device)
            z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(self.device).clamp(-0.5,0.5)
        a = F.relu(self.d1(torch.cat((state, z), 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))



class BCQ(object):
    def __init__(self, state_dim, action_dim, action_space, device, discount=0.90, tau=0.005, lmbda=0.75, phi=0.05):
        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, action_space.high, device, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        # self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        # self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1e-3)

        self.vae = VAE(state_dim, action_dim, latent_dim, action_space.high, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())
        # self.vae_optimizer = torch.optim.AdamW(self.vae.parameters())


        self.max_action = action_space.high
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device

        # self.vae_scaler = torch.cuda.amp.GradScaler()
        # self.actor_scaler = torch.cuda.amp.GradScaler()
        # self.critic_scaler = torch.cuda.amp.GradScaler()


    def select_action(self, state, eval=True):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(0)
            # ind = self.critic(state, action).max(0)[1]
        # return action[ind].cpu().data.numpy().flatten()
        return action[ind].cpu().data.numpy().flatten()

    def repeat_per_row(self, a, dim, n_repeats):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_repeats
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_repeats) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)

    def normalize_state(self, state, buff):
        return (state - buff.state_mean)/buff.state_std

    def unnormalize_state(self, state, buff):
        return state * buff.state_std + buff.state_mean

    def normalize_action(self, action, buff):
        return (action - buff.action_mean)/buff.action_std

    def unnormalize_action(self, action, buff):
        return action * buff.action_std + buff.action_mean

    def select_actions(self, states):
        with torch.no_grad():
            # states = torch.FloatTensor(states).to(self.device)
            # states = torch.FloatTensor(states).repeat(10, 1).to(self.device)
            # states = self.repeat_per_row(torch.FloatTensor(states), dim=0, n_repeats=10).to(self.device)
            # states = self.normalize_state(states, buff)

            states = torch.repeat_interleave(torch.FloatTensor(states), repeats=10, dim=0).to(self.device)


            # print("States", states)
            # print("States size", states.size())

            actions = self.actor(states, self.vae.decode(states))
            # actions = actions * buff.action_std + buff.action_mean

            # print("Actions", actions)
            # print("Actions size", actions.size())
            #
            q = self.critic.q1(states, actions)
            # print("q", q)
            # print("q size", q.size())

            q = torch.reshape(q, [-1, 10])

            # print("q", q)
            # print("q size", q.size())
            indices = q.argmax(1)
            # print("indices", q)
            # print("indices size", q.size())
            #
            # print("selected actions size", actions[indices].size())
            return actions[indices].cpu()
            # return actions.cpu()


        #     states = torch.FloatTensor(states).repeat(100, 1).to(self.device)
        #
        #     # states = torch.FloatTensor(states).to(self.device)
        #     print("States", states)
        #     print("States size", states.size())
        #     actions = self.actor(states, self.vae.decode(states))
        #     print("Actions", actions)
        #     print("Actions size", actions.size())
        #     q1 = self.critic.q1(states, actions)
        #     print("q1", q1)
        #     ind = q1.argmax(1)
        #     print("ind", ind)
        #     print("return", actions[ind].cpu().data.numpy().flatten())
        # return actions[ind].cpu().data.numpy().flatten()
        # return actions.cpu()



    def get_value_estimate(self, state):
        with torch.no_grad():
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            #ind = q1.argmax(0)
        return q1.cpu().data.numpy()


    def train(self, replay_buffer, iterations, batch_size=100, random_s=False, disable_tqdm=True):

        for it in tqdm(range(iterations), disable=disable_tqdm):
            # Sample replay buffer / batch

            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size, random_s=random_s, to_device=True) #O

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            # if optimize:
            vae_loss.backward()
            self.vae_optimizer.step()
                # self.vae_scaler.scale(vae_loss).backward()

                # self.vae_scaler.step(self.vae_optimizer)
                # self.vae_scaler.update()

            # Critic Training
            with torch.no_grad():
                # Duplicate next state 10 times
                next_state = torch.repeat_interleave(next_state, 10, 0).to(self.device)

                # Compute value of perturbed actions sampled from the VAE
                target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

                # Soft Clipped Double Q-learning
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
                # Take max over each action sampled from the VAE
                target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            # if optimize:
            critic_loss.backward()
            self.critic_optimizer.step()
                # self.critic_scaler.scale(critic_loss).backward()

                # self.critic_scaler.step(self.critic_optimizer)
                # self.critic_scaler.update()

            # Pertubation Model / Action Training
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)

            # Update through DPG
            actor_loss = -self.critic.q1(state, perturbed_actions).mean()

            self.actor_optimizer.zero_grad()
            # if optimize:
            actor_loss.backward()
            self.actor_optimizer.step()
                # self.actor_scaler.scale(actor_loss).backward()

                # self.actor_scaler.step(self.actor_optimizer)
                # self.actor_scaler.update()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # return actor_loss.detach().cpu().numpy().mean(), critic_loss.detach().cpu().numpy().mean(), vae_loss.detach().cpu().numpy().mean()

    def save(self, filename, type):
        torch.save(self.critic.state_dict(), filename + f"_bcq_critic_{type}")
        torch.save(self.critic_optimizer.state_dict(), filename + f"_bcq_critic_optimizer_{type}")

        torch.save(self.actor.state_dict(), filename + f"_bcq_actor_{type}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_bcq_actor_optimizer_{type}")

        torch.save(self.vae.state_dict(), filename + f"_bcq_vae_{type}")
        torch.save(self.vae_optimizer.state_dict(), filename + f"_bcq_vae_optimizer_{type}")


    def load(self, filename, type):
        self.critic.load_state_dict(torch.load(filename + f"_bcq_critic_{type}", map_location=torch.device(self.device)))
        self.critic_optimizer.load_state_dict(torch.load(filename + f"_bcq_critic_optimizer_{type}", map_location=torch.device(self.device)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + f"_bcq_actor_{type}", map_location=torch.device(self.device)))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_bcq_actor_optimizer_{type}", map_location=torch.device(self.device)))
        self.actor_target = copy.deepcopy(self.actor)

        self.vae.load_state_dict(torch.load(filename + f"_bcq_vae_{type}", map_location=torch.device(self.device)))
        self.vae_optimizer.load_state_dict(torch.load(filename + f"_bcq_vae_optimizer_{type}", map_location=torch.device(self.device)))
