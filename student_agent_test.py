import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from torchvision import transforms as T

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# 改成你自己存 DuelingCNN 的路徑
from training.rainbow import DuelingCNN  

CHECKPOINT_PATH = 'checkpoints/rainbow_5/rainbow_dqn_mario.pth'

class InferenceAgent:
    def __init__(self):
        # 1) 装置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2) 内部只做 JoypadSpace，其他 wrapper 手动实现
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)

        # 3) 跳帧参数
        self.skip = 4

        # 4) 重现 GrayScaleResize
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 90)),
            T.ToTensor(),    # → [1,84,90]
        ])

        # 5) FrameStack 大小
        self.k = 4
        self.frames = deque(maxlen=self.k)

        # 6) 载入 DuelingCNN
        n_actions = len(COMPLEX_MOVEMENT)
        self.model = DuelingCNN(in_channels=self.k, n_actions=n_actions).to(self.device)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(ckpt.get('model', ckpt))
        self.model.eval()

        # 7) 标记是否已做过内部 reset+prime
        self._primed = False

    def act(self, observation):
        """
        只吃原始 RGB frame (240×256×3, uint8)。
        内部自动完成：
          - 初次调用时 reset 内部 env + prime frame‑stack
          - 随后每步的 skip-frame、灰度→resize、4 帧 stack
          - 模型推断、更新 frame‑stack
        返回：
          next_raw_obs (240×256×3 uint8),
          total_reward (float),
          done (bool),
          info (dict)
        """
        # 1) 去负 strides
        raw = np.ascontiguousarray(observation)

        # 2) 如果还没 prime，就用这个 raw 做 prime，也 reset 内部 env
        if not self._primed:
            self.env.reset()
            proc0 = self.transform(raw)
            for _ in range(self.k):
                self.frames.append(proc0)
            self._primed = True

        # 3) 拼 state → [1,4,84,90]
        state = torch.cat(list(self.frames), dim=0)\
                     .unsqueeze(0)\
                     .to(self.device)

        # 4) forward & 选 action
        with torch.no_grad():
            q = self.model(state)
        action = int(q.argmax(dim=1).item())

        # 5) SkipFrame: 用内部 env.step 执行 skip 次
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.skip):
            try:
                next_raw, r, done, info = self.env.step(action)
                # 6) 更新 frame‑stack
                proc = self.transform(np.ascontiguousarray(next_raw))
                self.frames.append(proc)

            except ValueError:
                    # 环境已经 done，直接跳出
                    done = True
                    
                    break
        
            total_reward += r
            if done:
                break

        
        return action

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    agent = InferenceAgent()

    for ep in range(1, 11):
        obs = env.reset()     # raw RGB frame
        # agent.reset()         # 清空 frame‑stack
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(obs)           # act 只吃 raw obs
            obs, reward, done, info = env.step(action)
            total_reward += reward

        print(f'Episode {ep:2d} → total env reward: {total_reward:.2f}')

    env.close()