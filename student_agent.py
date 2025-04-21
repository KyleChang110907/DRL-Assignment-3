import torch
import torch.nn.functional as F
from training.rainbow import make_env, DuelingCNN  # 假設 DuelingCNN 與 make_env 在同一個檔案或同個套件
import numpy as np

CHECKPOINT_PATH = 'checkpoints/rainbow_5/rainbow_dqn_mario.pth'

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
