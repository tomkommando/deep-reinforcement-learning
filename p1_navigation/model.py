# We implement Dueling DQN and standard DQN here. See https://arxiv.org/pdf/1511.06581.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0, fc1_out_size=64, fc2_out_size=64, duel=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int)    :   Dimension of each state
            action_size (int)   :   Dimension of each action
            seed (int)          :   Random seed
            fc1_out_size (int)  :   Number of output nodes from layer 1
            fc2_out_size (int)  :   Number of output nodes from layer 2
            duel (bool)         :   If True, implement Dueling DQN training, otherwise DQN training
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.duel = duel

        # stream 1: state value estimation V(s)
        self.state_value = nn.Sequential(
            nn.Linear(state_size, fc1_out_size),
            nn.ReLU(),
            nn.Linear(fc1_out_size, fc2_out_size),
            nn.ReLU(),
            nn.Linear(fc2_out_size, 1)
        )

        # stream 2: estimate of advantage of each action A(s, a)
        self.advantage_value = nn.Sequential(
            nn.Linear(state_size, fc1_out_size),
            nn.ReLU(),
            nn.Linear(fc1_out_size, fc2_out_size),
            nn.ReLU(),
            nn.Linear(fc2_out_size, action_size)
        )

    def forward(self, state):
        """
            Return Q value for agent. See comments for details.
        """
        # calculate advantage value.
        advantage_value = self.advantage_value(state)

        # If using Dueling DQN, return Q = V(s) + ( A(s, a) - mean[ A(s,a) ] )
        if self.duel:
            state_value = self.state_value(state)
            return state_value + (advantage_value - advantage_value.mean())
        # Return only A(s,a) otherwise
        else:
            return advantage_value

