# inference_agent.py

import torch
import numpy as np
from collections import deque
from torchvision import transforms as T
from PIL import Image

# 请确保下面这两个类已经和训练时完全一样地定义过，或者从你的训练代码里 import 进来
from training.rainbow_icm import DuelingCNN, COMPLEX_MOVEMENT, SKIP_FRAMES, ICM

# -----------------------------
# 6. Inference Agent for ICM-DQN
# -----------------------------
class InferenceAgent:
    def __init__(self, model_path: str, device: str='cpu', skip: int=SKIP_FRAMES):
        # 1) 设备 & 网络加载
        self.device = torch.device(device)
        self.net = DuelingCNN(in_c=4, n_actions=len(COMPLEX_MOVEMENT))
                # 1) 设备 & 网络加载
        self.device = torch.device(device)
        self.net = DuelingCNN(in_c=4, n_actions=len(COMPLEX_MOVEMENT))
        # ICM is not needed for inference action selection, so we instantiate but do not load weights
        self.icm = ICM(feat_dim=self.net.features(torch.zeros(1,4,84,90)).shape[1],
                       n_actions=len(COMPLEX_MOVEMENT)).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(ckpt['model'])
        # self.icm.load_state_dict(ckpt.get('icm_opt', {}))  # if needed
        self.net.to(self.device).eval()
        # 2) 同样预处理
        self.transform = T.Compose([
            T.ToPILImage(), T.Grayscale(), T.Resize((84,90)), T.ToTensor()
        ])
        # 3) 帧堆叠 & 跳帧
        self.frames = deque(maxlen=4)
        self.skip_frames = skip-1
        self.skip_count = 0
        self.last_action = 0
        self.first = True
    def act(self, raw_obs: np.ndarray) -> int:
        # 预处理
        proc = self.transform(raw_obs).squeeze(0).numpy()
        if self.first:
            self.frames.clear()
            for _ in range(4): self.frames.append(proc)
            self.first = False
        if self.skip_count>0:
            self.skip_count -=1
            return self.last_action
        self.frames.append(proc)
        state = np.stack(self.frames,axis=0)
        st_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): q = self.net(st_t)
        action = int(q.argmax(1).item())
        self.last_action = action
        self.skip_count = self.skip_frames
        return action