import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import multiprocessing as mp
logger = mp.log_to_stderr()
logger.setLevel(logging.DEBUG)


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.continuos_action_dim = action_dim-1
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_continuos = nn.Linear(hidden2, self.continuos_action_dim )#continuos action head
        self.fc_bool = nn.Linear(hidden2, 1)        #bool action head

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.fc1(x))

        out = self.relu(self.fc2(out))
        out_continuos = self.tanh(self.fc_continuos(out))
        out_bool = (self.sigmoid(self.fc_bool(out)) > 0.5).float()
        out = torch.cat([out_continuos,out_bool],axis=-1)
        return out
        


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1+action_dim, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out
