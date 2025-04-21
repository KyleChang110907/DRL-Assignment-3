# inference_agent.py

import torch
import numpy as np
from collections import deque
from torchvision import transforms as T
from PIL import Image
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# 请确保下面这两个类已经和训练时完全一样地定义过，或者从你的训练代码里 import 进来
from training.rainbow import DuelingCNN

class InferenceAgent:
    def __init__(self, model_path: str, device: str = 'cpu'):
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
    agent = InferenceAgent('checkpoints/rainbow_5/rainbow_dqn_mario.pth', device='cpu')

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

        print(f"Episode {ep+1:2d} → reward = {total:6.1f}")
        rewards.append(total)

    print(f"Average reward: {np.mean(rewards):6.1f}")