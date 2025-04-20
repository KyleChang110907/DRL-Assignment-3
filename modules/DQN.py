import torch
from torch import nn
from modules.exploration import NoisyLinear
# ---------- Dueling DQN ----------
class DuelingDQN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(8, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        self.fc_input_dim = 64 * 7 * 7  # (84→20 after conv stack)
        # Advantage & value streams with noisy layers
        self.advantage = nn.Sequential(
            NoisyLinear(self.fc_input_dim, 512), nn.ReLU(),
            NoisyLinear(512, action_dim)
        )
        self.value = nn.Sequential(
            NoisyLinear(self.fc_input_dim, 512), nn.ReLU(),
            NoisyLinear(512, 1)
        )

    def forward(self, x):
        x = x / 255.0  # normalize pixel
        x = self.features(x).view(-1, self.fc_input_dim)
        adv, val = self.advantage(x), self.value(x)
        return val + adv - adv.mean(1, keepdim=True)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()