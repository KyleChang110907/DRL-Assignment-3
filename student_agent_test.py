# inference_agent.py

import torch
import gym
import numpy as np
from collections import deque
from torchvision import transforms as T

# 从训练脚本中导入网络结构
from training.rainbow import DuelingCNN

CHECKPOINT_PATH   = 'checkpoints/rainbow/rainbow_dqn_mario.pth'
N_STACKED_FRAMES  = 4
N_ACTIONS         = 12
DEVICE            = torch.device("cpu")

class Agent(object):
    """
    Inference Agent：自动完成从原始 RGB frame 到动作选择的完整流程。
    __init__ 和 act 签名与你的要求一致。
    """
    def __init__(self):
        # 1. 构建网络
        self.net = DuelingCNN(N_STACKED_FRAMES, N_ACTIONS).to(DEVICE)

        # 2. 加载 checkpoint，兼容两种格式：
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        self.net.load_state_dict(state_dict)
        self.net.eval()

        # 3. 预处理流水线：RGB → 灰度 → Resize → Tensor([1,84,90])
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),       # 1 通道
            T.Resize((84, 90)),  # H=84, W=90
            T.ToTensor()         # → tensor shape [1,84,90], dtype=float32
        ])

        # 4. 帧缓冲，用于堆叠
        self.frames = deque(maxlen=N_STACKED_FRAMES)

        # 5. 构造 gym 兼容的 action_space（可选）
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    def act(self, observation):
        """
        参数:
            observation: 原始环境输出的 RGB frame（H×W×3 的 numpy array）
                         或者已经预处理成 tensor[F,C,H,W] 也能兼容（会跳过 transform）
        返回:
            action: int
        """
        # 如果是 numpy 或 PIL，就做 transform；若是已经 Tensor，就假设它是 [1,84,90]
        if isinstance(observation, np.ndarray):
            obs_t = self.transform(observation)  # [1,84,90]
        elif torch.is_tensor(observation):
            obs_t = observation.float()
            # 确保 shape 是 [1,84,90] 而不是带 batch 的 [B,1,84,90]
            if obs_t.ndim == 4:
                obs_t = obs_t.squeeze(0)
        else:
            raise TypeError(f"Unsupported observation type: {type(observation)}")

        # 初始化时用同一帧填满 deque
        if len(self.frames) == 0:
            for _ in range(N_STACKED_FRAMES):
                self.frames.append(obs_t)
        else:
            self.frames.append(obs_t)

        # 拼成 [4,84,90]
        state = torch.cat(list(self.frames), dim=0)
        # 加 batch 维度 → [1,4,84,90]
        state = state.unsqueeze(0).to(DEVICE)

        # 前向并选最大 Q 的动作
        with torch.no_grad():
            q_vals = self.net(state)  # [1, N_ACTIONS]
        action = int(q_vals.argmax(dim=1).item())
        return action

