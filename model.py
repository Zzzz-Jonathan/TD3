import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, act_size), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(64 + act_size, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))
