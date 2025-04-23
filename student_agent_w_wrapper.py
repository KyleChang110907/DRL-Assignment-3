import torch
import torch.nn.functional as F
from training.rainbow import make_env, DuelingCNN  # 假設 DuelingCNN 與 make_env 在同一個檔案或同個套件
import numpy as np

CHECKPOINT_PATH = 'checkpoints/rainbow_11/rainbow_dqn_mario.pth'

class Agent:
    def __init__(self):
        # 1. 設定裝置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # 2. 用暫時環境取得 obs_shape 與 action 數量
        env = make_env()
        obs_shape = env.observation_space.shape  # e.g. (4,84,90)
        n_actions = env.action_space.n
        env.close()
        # 3. 初始化網路並載入權重
        self.model = DuelingCNN(obs_shape[0], n_actions).to(self.device)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device)
        # 如果你在存檔時將模型 state_dict 包在 'model' key 中，否則直接 load ckpt
        self.model.load_state_dict(ckpt.get('model', ckpt))
        self.model.eval()

    def act(self, observation):
        """
        根據原始 observation 回傳一個 action。
        observation: numpy array, shape 與訓練時相同 (例如 (4,84,90))
        """
         # 1) 確保 numpy 陣列是 contiguous，去除負 strides
        obs = np.ascontiguousarray(observation)
        # 2) 轉成 tensor
        state = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

        # # 1. 轉 tensor 並送到裝置
        # state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        # 2. 前向推論
        with torch.no_grad():
            q_vals = self.model(state)
        # 3. 選取最大 Q-value 的動作
        return int(q_vals.argmax(dim=1).item())

# import torch
# import torch.nn.functional as F
# import numpy as np
# from collections import deque
# from torchvision import transforms as T
# from nes_py.wrappers import JoypadSpace
# from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# from training.rainbow import DuelingCNN  # 請換成你實際的 import 路徑

# CHECKPOINT_PATH = 'checkpoints/rainbow_5/rainbow_dqn_mario.pth'

# class Agent:
#     def __init__(self):
#         # 裝置
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # 重現訓練時的 GrayScaleResize transform
#         self.transform = T.Compose([
#             T.ToPILImage(),
#             T.Grayscale(),        # output channel = 1
#             T.Resize((84, 90)),   # H=84, W=90
#             T.ToTensor()          # → tensor shape [1,84,90], range [0,1]
#         ])

#         # frame-stack 容量
#         self.k = 4
#         self.frames = deque(maxlen=self.k)

#         # 載入模型（in_channels=4, n_actions 同你的 COMPLEX_MOVEMENT）
#         n_actions = len(COMPLEX_MOVEMENT)
#         self.model = DuelingCNN(in_channels=self.k, n_actions=n_actions).to(self.device)
#         ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device)
#         self.model.load_state_dict(ckpt.get('model', ckpt))
#         self.model.eval()

#     def act(self, observation):
#         """
#         支持兩種 observation：
#          - 原始 RGB frame: np.ndarray, shape=(240,256,3), dtype=uint8
#          - 已包裝後: np.ndarray, shape=(4,84,90), dtype=float32, range [0,1]
#         回傳 int 動作。
#         """
#         # 1) 確保 contiguous 以消除負 strides
#         # print(f"observation shape: {observation.shape}, dtype: {observation.dtype}")
#         obs = np.ascontiguousarray(observation)

#         # 2) 判斷是不是已經是 (4,84,90) 的 preprocessed stack
#         if obs.ndim == 3 and obs.shape == (self.k, 84, 90):
#             # 直接當 state
#             state = torch.from_numpy(obs).unsqueeze(0).to(self.device).float()
#             # print(f"state shape: {state.shape}, dtype: {state.dtype}")
#         else:
#             # 原始 RGB frame → 灰度、resize、stack
#             gray = self.transform(obs)  # [1,84,90], float32
#             if len(self.frames) < self.k:
#                 # 開頭用同一張複製 k 次
#                 for _ in range(self.k):
#                     self.frames.append(gray)
#             else:
#                 self.frames.append(gray)
#             # 拼成 [4,84,90] 再加 batch dim
#             state = torch.cat(list(self.frames), dim=0)\
#                         .unsqueeze(0)\
#                         .to(self.device)
#         # print(f'state:\n',state)
#         # assert False

#         # 3) forward（訓練時在 model 裡已做 x/255.0）
#         with torch.no_grad():
#             q_vals = self.model(state)
#         return int(q_vals.argmax(dim=1).item())


