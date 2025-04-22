# inference_agent.py

import torch
import numpy as np
from collections import deque
from torchvision import transforms as T
from PIL import Image
import cv2
import time
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# 请确保下面这两个类已经和训练时完全一样地定义过，或者从你的训练代码里 import 进来
from training.rainbow import DuelingCNN
from training.rainbow2_sf import SKIP_FRAMES

class InferenceAgent:
    def __init__(self, model_path: str, device: str = 'cpu'):
        # 1. 设备 & 网络
        self.obs_c, self.h, self.w = 4, 84, 90
        self.n_actions = len(COMPLEX_MOVEMENT)
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
        self.last_state = None

        # **新增：跳帧管理**
        self.skip_count = 0
        self.last_action = 0

        # 用来检测 reset
        self.first_raw       = None
        self.reset_threshold = 5.0   # 平均像素差阈值       
    def _process(self, raw):
        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img = resized.astype(np.float32)
        self.frames.append(img)
        return np.stack(self.frames, axis=0)
    
    def act(self, raw_obs: np.ndarray) -> int:
        """
        输入:
          raw_obs (H×W×3) uint8 RGB numpy array
        返回:
          action (int)
        """
        # —— 1. 首次调用时记录第一帧
        if self.first_raw is None:
            self.first_raw = raw_obs.copy()

        # —— 2. 检测是否为 reset（done）
        # —— 2. 如果当前帧和第一帧【完全相同】，就认为是刚 reset
        if np.array_equal(raw_obs, self.first_raw):
            self.frames.clear()
            self.skip_count = 0
            self.last_action = 0
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # print(f"{current_time} | Environment reset detected.")


        # 先用新帧更新 frame‑stack
        if len(self.frames) == 0:
            gray = cv2.cvtColor(raw_obs, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
            img = resized.astype(np.float32)
            for _ in range(self.obs_c):
                self.frames.append(img)
        state = self._process(raw_obs)
        self.last_state = state

        # 如果现在还在跳帧期，就直接返回上次的动作
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        # 否则真正跑网络选新动作
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.net(tensor)
        action = int(q.argmax(1).item())

        # 记录，并重置跳帧计数
        self.last_action = action
        self.skip_count = SKIP_FRAMES - 1
        return action

if __name__ == "__main__":
    MAX_EPISODE_STEPS = 3000
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym.wrappers import TimeLimit
    
    # 1. 创建“只能这么长” 的测试环境
    env   = gym_super_mario_bros.make('SuperMarioBros-v0')
    env   = JoypadSpace(env, COMPLEX_MOVEMENT)
    env   = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

    # 2. 实例化推理 Agent（载入你训练好的 checkpoint）
    agent = InferenceAgent('checkpoints/rainbow_23/rainbow_mario.pth', device='cpu')

    # 3. 跑十集
    rewards = []
    for ep in range(10):
        obs   = env.reset()   # raw RGB frame, shape=(240,256,3)
        done  = False
        total = 0
        steps = 0

        # 注意：第一次调用 act() 时内部队列会自动填满；不用额外清零
        while not done and steps < 2000:
            # 只用原图调用 act，内部自动做所有 wrapper 操作
            a, steps = agent.act(obs), steps + 1

            obs, r, done, info = env.step(a)
            total += r
            env.render()
            time.sleep(0.01)
        print(f"Episode {ep+1:2d} → reward = {total:6.1f}")
        rewards.append(total)

    print(f"Average reward: {np.mean(rewards):6.1f}")