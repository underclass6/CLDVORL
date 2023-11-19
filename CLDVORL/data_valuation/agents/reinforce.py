""" Data Valuation based Batch-Constrained Reinforcement Learning
    -------------------------------------------------------------
    REINFORCE agent that is used as a data value estimator. """

from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class REINFORCE(nn.Module):

    def __init__(self, state_dim, action_dim, layers=[128, 128], args={}):
        super(REINFORCE, self).__init__()
        # policy network
        self.fc1 = nn.Linear(state_dim, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], action_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
