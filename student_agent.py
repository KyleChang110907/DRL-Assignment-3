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
from torchvision import transforms as T
from collections import deque

from training.rainbow import DuelingCNN, COMPLEX_MOVEMENT, make_env, SKIP_FRAMES
model_path = 'checkpoints/rainbow_11/rainbow_dqn_mario.pth'
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

        # 4. 自訂的環境
        self.env = make_env()
        self.state = self.env.reset()

        # 5. skip_frames
        self.skip_count = 0
        self.last_action = 0

        # 6. 用來檢測 reset
        self.first_raw = None
        self.done = False


    def act(self, raw_obs: np.ndarray) -> int:
        """
        输入:
          raw_obs (H×W×3) uint8 RGB numpy array
        返回:
          action (int)
        """
        # —— 1. 预处理当前帧 —— 
        # transform -> tensor [1,84,90]
        if self.first_raw is None:
            self.first_raw = raw_obs.copy()
        
        if np.array_equal(raw_obs, self.first_raw):
            print('done while same as first frame')
            self.frames.clear()
            self.skip_count = 0
            self.last_action = 0
            self.state = self.env.reset()
            self.done = False

        t = self.transform(raw_obs)  
        # 转成 numpy [84,90]
        f = t.squeeze(0).numpy()      

        # —— 2. 如果是新一集, 先把队列填满同一帧 —— 
        if len(self.frames) == 0:
            for _ in range(4):
                self.frames.append(f)
        else:
            self.frames.append(f)

        if self.done and self.skip_count == 0:
            print('done')
            self.frames.clear()
            self.skip_count = 0
            self.last_action = 0
            self.state = self.env.reset()
            self.done = False

        elif self.done and self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action
        
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action
        
        state = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.net(state)
            action = int(q.argmax(1).item())
            self.last_action = action
            self.skip_count = 0

        self.state, env_r, self.done, info = self.env.step(self.last_action)
        self.skip_count = self.state.shape[0] - 1

        if self.done:
            print(f'done within {self.skip_count} frames')

        return action
