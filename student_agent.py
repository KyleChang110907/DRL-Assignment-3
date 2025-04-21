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
