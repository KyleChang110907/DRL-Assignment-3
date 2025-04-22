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
CHECKPOINT_PATH   = 'checkpoints/rainbow_23/rainbow_mario.pth'

from training.rainbow2_sf import DuelingCNN, SKIP_FRAMES

# -----------------------------
# Inference‑only Agent
# -----------------------------
class InferenceAgent:
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

        # **新增：跳帧管理**
        self.skip_count = 0
        self.last_action = 0

        # 用来检测 reset
        self.first_raw       = None
        self.reset_threshold = 5.0   # 平均像素差阈值

    def _process(self, raw):
        gray    = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img     = resized.astype(np.float32)
        self.frames.append(img)
        return np.stack(self.frames, axis=0)

    def act(self, raw):
        # —— 1. 首次调用时记录第一帧
        if self.first_raw is None:
            self.first_raw = raw.copy()

        # —— 2. 检测是否为 reset（done）
        # —— 2. 如果当前帧和第一帧【完全相同】，就认为是刚 reset
        if np.array_equal(raw, self.first_raw):
            self.frames.clear()
            self.skip_count = 0
            self.last_action = 0
            # current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # print(f"{current_time} | Environment reset detected.")


        # 先用新帧更新 frame‑stack
        if len(self.frames) == 0:
            gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
            img = resized.astype(np.float32)
            for _ in range(self.obs_c):
                self.frames.append(img)
        state = self._process(raw)
        self.last_state = state

        # 如果现在还在跳帧期，就直接返回上次的动作
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        # 否则真正跑网络选新动作
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model(tensor)
        action = int(q.argmax(1).item())

        # 记录，并重置跳帧计数
        self.last_action = action
        self.skip_count = SKIP_FRAMES - 1
        return action

# -----------------------------
# Inference Loop
# -----------------------------
def run_inference(num_episodes=5, render=True):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

    agent = InferenceAgent()

    for ep in range(1, num_episodes + 1):
        obs   = env.reset()
        done  = False
        total = 0.0

        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total += reward
            if render:
                env.render()

        print(f"Episode {ep:>2d}  Raw Reward: {total:.2f}")
    env.close()

if __name__ == '__main__':
    run_inference(num_episodes=10, render=True)
