# inference_mario.py

import os
import cv2
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# -----------------------------
# Hyperparameters / paths
# -----------------------------
MAX_EPISODE_STEPS = 2000
CHECKPOINT_PATH   = 'checkpoints/rainbow_21/rainbow_mario.pth'

from training.rainbow2 import DuelingCNN

# -----------------------------
# Inference‑only Agent
# -----------------------------
class Agent:
    def __init__(self):
        # same obs shape as training
        self.obs_c, self.h, self.w = 4, 84, 90
        self.n_actions = len(COMPLEX_MOVEMENT)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load network
        self.model = DuelingCNN(self.obs_c, self.n_actions).to(self.device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()  # deterministic, no noise

        # frame buffer
        self.frames = deque(maxlen=self.obs_c)
        self.last_state = None

    def _process(self, raw):
        gray    = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img     = resized.astype(np.float32)
        self.frames.append(img)
        return np.stack(self.frames, axis=0)

    def act(self, raw):
        # on first call (or after clear), fill deque with identical frames
        if len(self.frames) == 0:
            gray    = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
            img     = resized.astype(np.float32)
            for _ in range(self.obs_c):
                self.frames.append(img)

        state = self._process(raw)
        self.last_state = state
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(tensor)
        return int(q.argmax(1).item())

# # -----------------------------
# # Inference Loop
# # -----------------------------
# def run_inference(num_episodes=5, render=True):
#     env = gym_super_mario_bros.make('SuperMarioBros-v0')
#     env = JoypadSpace(env, COMPLEX_MOVEMENT)
#     env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

#     agent = Agent()

#     for ep in range(1, num_episodes + 1):
#         obs   = env.reset()
#         done  = False
#         total = 0.0

#         while not done:
#             action = agent.act(obs)
#             obs, reward, done, info = env.step(action)
#             total += reward
#             if render:
#                 env.render()

#         print(f"Episode {ep:>2d}  Raw Reward: {total:.2f}")
#     env.close()

# if __name__ == '__main__':
#     run_inference(num_episodes=10, render=True)

import torch
import numpy as np
from collections import deque
from torchvision import transforms as T
from PIL import Image
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# 请确保下面这两个类已经和训练时完全一样地定义过，或者从你的训练代码里 import 进来
from training.rainbow import DuelingCNN
model_path = 'checkpoints/rainbow_5/rainbow_dqn_mario.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Agent:
    def __init__(self):
        # 1. 设备 & 网络
        self.device = torch.device(device)
        # 假设训练时网络接收 4 帧、输出 len(COMPLEX_MOVEMENT) 个动作
        self.net = DuelingCNN(in_channels=4, n_actions=len(COMPLEX_MOVEMENT))
        # 加载权重
        ckpt = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        self.net.to(self.device).eval()

        # 2. 完全复刻训练时的预处理
        self.transform = T.Compose([
            T.ToPILImage(),           # numpy( H×W×3 ) -> PIL
            T.Grayscale(),            # -> 1×H×W
            T.Resize((84, 90)),       # -> 1×84×90
            T.ToTensor(),             # -> tensor [1,84,90], float32 in [0,1]
        ])

        # 3. 帧堆叠队列
        self.frames = deque(maxlen=4)

    def act(self, raw_obs: np.ndarray) -> int:
        """
        输入:
          raw_obs (H×W×3) uint8 RGB numpy array
        返回:
          action (int)
        """
        # —— 1. 预处理当前帧 —— 
        # transform -> tensor [1,84,90]
        t = self.transform(raw_obs)  
        # 转成 numpy [84,90]
        f = t.squeeze(0).numpy()      

        # —— 2. 如果是新一集, 先把队列填满同一帧 —— 
        if len(self.frames) == 0:
            for _ in range(4):
                self.frames.append(f)
        else:
            self.frames.append(f)

        # —— 3. 堆成(4×84×90) 送入网络 —— 
        state = np.stack(self.frames, axis=0)           # shape = (4,84,90)
        st_t  = torch.from_numpy(state).unsqueeze(0)    # shape = (1,4,84,90)
        st_t  = st_t.to(self.device, dtype=torch.float32)

        # —— 4. 推理并选动作 —— 
        with torch.no_grad():
            q = self.net(st_t)           # shape = (1, n_actions)
            action = int(q.argmax(1).item())
        return action
