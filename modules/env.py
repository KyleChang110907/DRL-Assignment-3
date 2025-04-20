# modules/env.py
import gym
import gym_super_mario_bros as smb
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

import numpy as np
import torch
from PIL import Image

# ---------- 自行實作 SkipFrame ----------
class SkipFrame(gym.Wrapper):
    """Return only every `skip`-th frame (summing rewards)."""
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        frame = None
        for _ in range(self._skip):
            obs, reward, terminated, info = self.env.step(action)
            total_reward += reward
            frame = obs
            if terminated or truncated:
                break
        return frame, total_reward, terminated, info

# ---------- 建立環境: 僅做跳幀與按鍵空間映射 ----------
def make_env():
    env = smb.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    return env

# ---------- 單張畫面預處理 ----------
def preprocess(obs):
    """Convert raw RGB frame (H,W,3) or array-like to torch.uint8 tensor shape (84,84)."""
    # to numpy array
    arr = obs if isinstance(obs, np.ndarray) else np.array(obs)
    # convert to PIL, grayscale and resize
    img = Image.fromarray(arr)
    img = img.convert('L').resize((84, 84), resample=Image.BILINEAR)
    # to uint8 tensor
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr)  
