import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
import random
import numpy as np
from tqdm import tqdm
from CLDVORL.CLDV.replay_buffer import ReplayBuffer


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# classifier model
class Discriminator(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim, disc_obs_index=None):
        super(Discriminator, self).__init__()

        self.disc_obs_index = disc_obs_index
        if disc_obs_index is not None:
            num_inputs = len(disc_obs_index)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear31 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear32 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear33 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)

    def forward(self, state):
        if self.disc_obs_index is not None:
            state = state[:, self.disc_obs_index]
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # x = F.relu(self.linear33(F.relu(self.linear32(F.relu(self.linear31(x))))))
        x = self.linear4(x)
        x = 2 * F.tanh(x)  # TBD

        return x  # regression label, unnormalized


class DeltaCla(object):
    def __init__(self, state_dim, action_dim, args):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cla_sas = Discriminator(state_dim * 2 + action_dim, 2, args.dcla_hidden_size).to(device=self.device)
        self.cla_sa = Discriminator(state_dim + action_dim, 2, args.dcla_hidden_size).to(device=self.device)
        self.cla_sas_optim = RMSprop(self.cla_sas.parameters(), lr=args.dcla_lr)
        self.cla_sa_optim = RMSprop(self.cla_sa.parameters(), lr=args.dcla_lr)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def delta_dynamic(self, s, a, ss):
        sas = torch.cat([s, a, ss], 1)
        sa = torch.cat([s, a], 1)

        prob_sas = F.softmax(self.cla_sas(sas), dim=1).detach().cpu().numpy()

        delata_sas = np.log(prob_sas[:, 1]) - np.log(prob_sas[:, 0])

        prob_sa = F.softmax(self.cla_sa(sa), dim=1).detach().cpu().numpy()
        delata_sa = np.log(prob_sa[:, 1]) - np.log(prob_sa[:, 0])
        delta = delata_sas - delata_sa
        return np.clip(delta, -10, 10)


    def update_param_cla(self, data_s, data_t, batch_size):
        losses = 0
        losssas = 0
        losssa = 0
        for index in [0, 1]:
            if index == 0: data = data_s
            if index == 1: data = data_t

            state_batch, action_batch, next_state_batch, _, _ = data.sample(batch_size=batch_size, random_s=True)

            state_batch = state_batch.to(self.device)
            action_batch = action_batch.to(self.device)
            next_state_batch = next_state_batch.to(self.device)

            state_action_state_batch = torch.cat([state_batch, action_batch, next_state_batch], 1)
            state_action_batch = torch.cat([state_batch, action_batch], 1)
            sort_index = torch.tensor([index] * batch_size).to(self.device)
            cla_sas_loss = F.cross_entropy(self.cla_sas(state_action_state_batch), sort_index)

            cla_sa_loss = F.cross_entropy(self.cla_sa(state_action_batch), sort_index)


            self.cla_sas_optim.zero_grad()
            self.cla_sa_optim.zero_grad()
            cla_sas_loss.backward()
            cla_sa_loss.backward()
            self.cla_sas_optim.step()
            self.cla_sa_optim.step()
            losssas = losssas + cla_sas_loss.item()
            losssa = losssa + cla_sa_loss.item()
        losses = losssas + losssa
        return losses, losssas, losssa

    def change_device2cpu(self):
        self.cla_sas.cpu()
        self.cla_sa.cpu()

    def change_device2device(self):
        self.cla_sas.to(device=self.device)
        self.cla_sa.to(device=self.device)

    def train(self, source_data: ReplayBuffer, target_data: ReplayBuffer, args):
        print('Start training delta classifier...')

        for epoch in tqdm(range(args.dcla_epochs)):
            for e in range(args.updates_per_step_cla):
                cal_loss = self.update_param_cla(source_data, target_data, args.dcla_batch_size)
                print(cal_loss[0])

        print('Finish!')

    def save_delta_models(self, path):
        if os.path.exists(path+"/delta_models/"):
            torch.save(self.cla_sas.state_dict(), path + '/delta_models/cla_sas')
            torch.save(self.cla_sa.state_dict(), path + '/delta_models/cla_sa')
        else:
            os.makedirs(path+"/delta_models/")
            torch.save(self.cla_sas.state_dict(), path + '/delta_models/cla_sas')
            torch.save(self.cla_sa.state_dict(), path + '/delta_models/cla_sa')

    def load_delta_models(self, path):
        self.cla_sas.load_state_dict(torch.load(path + '/delta_models/cla_sas', map_location=torch.device(self.device)))
        self.cla_sa.load_state_dict(torch.load(path + '/delta_models/cla_sa', map_location=torch.device(self.device)))

