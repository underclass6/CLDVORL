
""" Data Valuation based Batch-Constrained Reinforcement Learning 
    -------------------------------------------------------------
    Given s, predict (s_, a, r). """

from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Predictor(nn.Module):

    def __init__(self, input_dim, action_dim, action_space, layers=[128, 128]):
        super(Predictor, self).__init__()
        # policy network
        self.fc1 = nn.Linear(input_dim+action_dim, layers[0])

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.new_state = nn.Linear(layers[1], input_dim)
        # self.actions = nn.Linear(layers[1], action_dim)
        self.reward = nn.Linear(layers[1], 1)
        self.terminals = nn.Linear(layers[1], 1)
        self.max_action = float(action_space.high[0])
        # self.max_action = torch.FloatTensor(max_action).to(device)


    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        s_ = self.new_state(x)
        r = self.reward(x)

        # a = self.actions(x)
        # a = self.tanh(a) * self.max_action

        t = self.terminals(x)
        t = self.sigmoid(t)
        # return s_, r, a, t
        return s_, r, t

