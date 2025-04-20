import torch
from torch import nn
import math
import torch.nn.functional as F
# ---------- Noisy Linear for exploration ----------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_mu   = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma= nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.bias_mu     = nn.Parameter(torch.empty(out_features))
        self.bias_sigma  = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.std_init = std_init
        self.reset_parameters(); self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        eps_in  = torch.randn(self.in_features, device=self.weight_eps.device)
        eps_out = torch.randn(self.out_features, device=self.weight_eps.device)
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)
