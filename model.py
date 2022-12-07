import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


class PolicyNet(nn.Module):
    def __init__(self, env):
        super(PolicyNet, self).__init__()
        self.env = env
        # network
        self.hidden = nn.Linear(40, 60)
        self.out = nn.Linear(60, 1)

        # training
        self.optimizer = optim.Adam(self.parameters(), lr=1)

    def forward(self, observation):
        x = torch.Tensor(observation).view(40)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x)
        x = torch.round(x)

        return x

    def loss(self, action_probabilities, reward):
        return -torch.mean(torch.mul(torch.log(action_probabilities), reward))