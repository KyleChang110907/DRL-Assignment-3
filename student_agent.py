# inference_agent.py

import torch
import gym
import numpy as np
from collections import deque
from torchvision import transforms as T

from training.rainbow import DuelingCNN

CHECKPOINT_PATH   = 'checkpoints/rainbow_3/rainbow_dqn_mario.pth'
N_STACKED_FRAMES  = 4
N_ACTIONS         = 12
DEVICE            = torch.device("cpu")
TEMPERATURE       = 2.0   # Softmax 温度，<1 更加贪心，>1 更加随机

class Agent(object):
    """
    Stochastic Inference Agent：使用 NoisyNet 噪声和 softmax sampling
    __init__ 与 act 签名保持不变。
    """
    def __init__(self):
        # 1. 建网络并加载权重
        self.net = DuelingCNN(N_STACKED_FRAMES, N_ACTIONS).to(DEVICE)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        self.net.load_state_dict(state_dict)

        # 2. **保持训练模式**，让 NoisyLinear 在 forward 时注入噪声
        self.net.train()

        # 3. 预处理流水线
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 90)),
            T.ToTensor()
        ])

        # 4. 帧缓冲
        self.frames = deque(maxlen=N_STACKED_FRAMES)

        # 5. action_space
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    def act(self, observation):
        # 1) 预处理到 [1,84,90]
        if isinstance(observation, np.ndarray):
            obs_t = self.transform(observation)
        elif torch.is_tensor(observation):
            obs_t = observation.float().squeeze(0)
        else:
            raise TypeError(f"Unsupported obs: {type(observation)}")

        # 2) 堆帧
        if not self.frames:
            for _ in range(N_STACKED_FRAMES):
                self.frames.append(obs_t)
        else:
            self.frames.append(obs_t)
        state = torch.cat(list(self.frames), dim=0).unsqueeze(0).to(DEVICE)  # [1,4,84,90]

        # 3) 重置噪声并前向
        self.net.reset_noise()
        with torch.no_grad():
            q_vals = self.net(state).squeeze(0)  # [N_ACTIONS]

        # 4) Softmax sampling
        probs = torch.softmax(q_vals / TEMPERATURE, dim=0)
        action = torch.multinomial(probs, num_samples=1).item()
        return action
