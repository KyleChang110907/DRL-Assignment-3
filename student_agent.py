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
model_path = 'checkpoints/rainbow_11/rainbow_dqn_mario_7918_backup.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Agent:
    def __init__(self):
        # —— 1. 设备 & 网络 —— 
        self.device = torch.device(device)
        self.net = DuelingCNN(in_channels=4, n_actions=len(COMPLEX_MOVEMENT))
        ckpt = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(ckpt.get('model', ckpt))
        self.net.to(self.device).eval()

        # —— 2. 训练时同样的预处理（灰度+resize）—— 
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 90)),
            T.ToTensor(),
        ])

        # —— 3. 帧堆叠队列 + 跳帧参数 —— 
        self.frames      = deque(maxlen=4)
        self.skip_frames = SKIP_FRAMES - 1   # e.g. skip=4 -> 中间 3 帧不选新动作
        self.skip_count  = 0
        self.last_action = 0
        self.first       = True

    def act(self, raw_obs: np.ndarray) -> int:
        """
        raw_obs: 外部 env.reset()/env.step() 给出的原图 (240×256×3)
        """

        # —— A. 预处理当前原始帧 —— 
        proc = self.transform(raw_obs).squeeze(0).numpy()  # shape=(84,90)

        # —— B. 第一次用此帧填满 deque —— 
        if self.first:
            self.frames.clear()
            for _ in range(4):
                self.frames.append(proc)
            self.first = False

        # —— C. 如果还在 skip 期间，直接返回 last_action —— 
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        # —— D. 更新帧队列，堆叠成 state —— 
        self.frames.append(proc)
        state = np.stack(self.frames, axis=0)           # (4,84,90)
        st_t  = torch.from_numpy(state).unsqueeze(0)    # (1,4,84,90)
        st_t  = st_t.to(self.device, dtype=torch.float32)

        # —— E. 网络推理选动作 —— 
        with torch.no_grad():
            q = self.net(st_t)
        action = int(q.argmax(1).item())
        self.last_action = action

        # —— F. 重置 skip 计数 —— 
        self.skip_count = self.skip_frames

        return action
