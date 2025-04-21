import os
import random
import datetime
from collections import deque, namedtuple

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import TimeLimit

import gym
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T

# -----------------------------
# Hyperparameters
# -----------------------------
COPY_NETWORK_FREQ       = 10000        # steps
BUFFER_CAPACITY         = 10000
BATCH_SIZE              = 32
GAMMA                   = 0.9
EPS_START               = 1.0
EPS_END                 = 0.01
EPS_DECAY               = 0.9999
LEARNING_RATE           = 0.00025
ADAM_EPS                = 0.00015
PER_ALPHA               = 0.6
PER_BETA_START          = 0.4
PER_BETA_FRAMES         = 2000000     # anneal β to 1.0
PER_EPSILON             = 0.1
N_STEP                  = 5
NOISY_SIGMA_INIT        = 2.5
# new hyperparameters
BACKWARD_PENALTY      = 0 #-1
STAY_PENALTY         = 0 #-0.2
DEATH_PENALTY        = -100 #-50


MAX_FRAMES              = 44800000    # total training frames

# 最大步数，超过便 truncated
MAX_EPISODE_STEPS = 3000

# -----------------------------
# 1. Environment Wrappers
# -----------------------------
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 定義 transform
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),        # output channel = 1
            T.Resize((84, 90)),   # H=84, W=90
            T.ToTensor()          # → tensor shape [1,84,90]
        ])
        # **同步更新 observation_space**
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(1, 84, 90),
            dtype=np.float32
        )

    def observation(self, obs):
        # input obs 是原始 RGB frame，轉成 [1,84,90] 的灰度 tensor
        return self.transform(obs)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0,
            shape=(shp[0]*k, shp[1], shp[2]),
            dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0), reward, done, info

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleResize(env)
    env = FrameStack(env, 4)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return env
    

# -----------------------------
# 2. Noisy Linear Layer
# -----------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=NOISY_SIGMA_INIT):
        super().__init__()
        self.in_f, self.out_f = in_features, out_features
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.in_f)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init / np.sqrt(self.in_f))
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_sigma, self.sigma_init / np.sqrt(self.out_f))

    def reset_noise(self):
        eps_in  = self._f(torch.randn(self.in_f))
        eps_out = self._f(torch.randn(self.out_f))
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

# -----------------------------
# 3. Dueling CNN
# -----------------------------
class DuelingCNN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 計算特徵維度
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 90)
            n_flat = self.features(dummy).shape[1]

        # Value stream
        self.val_noisy = NoisyLinear(n_flat, 512)
        self.val       = NoisyLinear(512, 1)
        # Advantage stream
        self.adv_noisy = NoisyLinear(n_flat, 512)
        self.adv       = NoisyLinear(512, n_actions)

    def forward(self, x):
        x = self.features(x / 255.0)
        v = F.relu(self.val_noisy(x))
        v = self.val(v)
        a = F.relu(self.adv_noisy(x))
        a = self.adv(a)
        return v + (a - a.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for m in [self.val_noisy, self.val, self.adv_noisy, self.adv]:
            m.reset_noise()

# -----------------------------
# 4. PER + N‑step Replay Buffer
# -----------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha, beta_start, beta_frames, n_step, gamma):
        self.cap = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta_by_frame = lambda f: min(1.0,
            beta_start + f * (1.0 - beta_start) / beta_frames)
        self.n_step = n_step
        self.gamma = gamma

        self.pos = 0
        self.buffer = []
        self.prios = np.zeros((capacity,), dtype=np.float32)
        self.n_buf = deque(maxlen=n_step)
        self.Exp = namedtuple('Exp', ['s','a','r','s2','d'])

    def _get_n_step(self):
        r, s2, d = self.n_buf[-1].r, self.n_buf[-1].s2, self.n_buf[-1].d
        for trans in reversed(list(self.n_buf)[:-1]):
            r = trans.r + self.gamma * r * (1 - trans.d)
            s2, d = (trans.s2, trans.d) if trans.d else (s2, d)
        return r, s2, d

    def add(self, s, a, r, s2, d):
        self.n_buf.append(self.Exp(s,a,r,s2,d))
        if len(self.n_buf) < self.n_step:
            return
        r_n, s2_n, d_n = self._get_n_step()
        s0, a0 = self.n_buf[0].s, self.n_buf[0].a
        exp = self.Exp(s0, a0, r_n, s2_n, d_n)
        if len(self.buffer) < self.cap:
            self.buffer.append(exp)
            prio = 1.0 if len(self.buffer) == 1 else self.prios.max()
        else:
            self.buffer[self.pos] = exp
            prio = self.prios.max()
            
        self.prios[self.pos] = prio
        self.pos = (self.pos + 1) % self.cap


    def sample(self, batch_size, frame_idx):
        N = len(self.buffer)
        if N == 0:
            return [], [], [], [], [], [], []

        # 1) 计算抽样概率
        prios = self.prios[:N] ** self.alpha
        sum_prios = prios.sum()
        if sum_prios <= 0:
            # 全部权重为零时，退化成均匀采样
            probs = np.ones_like(prios) / N
        else:
            probs = prios / sum_prios

        # 2) 按概率抽样
        idxs = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[i] for i in idxs]

        # 3) 计算 importance‑sampling 权重
        beta = self.beta_by_frame(frame_idx)
        weights = (N * probs[idxs]) ** (-beta)
        weights /= weights.max()

        batch = self.Exp(*zip(*samples))
        return (
            np.array(batch.s),
            batch.a,
            batch.r,
            np.array(batch.s2),
            batch.d,
            weights.astype(np.float32),
            idxs
        )


    def update_priorities(self, idxs, errors):
        for i,e in zip(idxs, errors):
            self.prios[i] = abs(e) + 1e-6

# -----------------------------
# 5. Agent
# -----------------------------
class Agent:
    def __init__(self, obs_shape, n_actions, device):
        self.device = device
        self.n_actions = n_actions

        # networks
        self.online = DuelingCNN(obs_shape[0], n_actions).to(device)
        self.target = DuelingCNN(obs_shape[0], n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.online.parameters(),
                              lr=LEARNING_RATE, eps=ADAM_EPS)

        self.buffer = PrioritizedReplayBuffer(
            capacity=BUFFER_CAPACITY,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START,
            beta_frames=PER_BETA_FRAMES,
            n_step=N_STEP,
            gamma=GAMMA
        )
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.frame_idx = 0
        self.update_freq = COPY_NETWORK_FREQ

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online(state)
        return int(q.argmax(1).item())

    def learn(self):
        if self.frame_idx < self.batch_size:
            return
        s,a,r,s2,d,w,idxs = self.buffer.sample(self.batch_size, self.frame_idx)

        s   = torch.tensor(s,  dtype=torch.float32).to(self.device)
        a   = torch.tensor(a).to(self.device)
        r   = torch.tensor(r,  dtype=torch.float32).to(self.device)
        s2  = torch.tensor(s2, dtype=torch.float32).to(self.device)
        d   = torch.tensor(d,  dtype=torch.float32).to(self.device)
        w   = torch.tensor(w,  dtype=torch.float32).to(self.device)

        # Double DQN target
        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        a_next = self.online(s2).argmax(1)
        q_next = self.target(s2).gather(1, a_next.unsqueeze(1)).squeeze(1)
        q_tar  = r + (self.gamma**N_STEP) * q_next * (1 - d)

        td = q_pred - q_tar.detach()
        loss = (F.smooth_l1_loss(q_pred, q_tar.detach(), reduction='none') * w).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.online.reset_noise()
        self.target.reset_noise()

        self.buffer.update_priorities(idxs, td.detach().cpu().numpy())

        if self.frame_idx % self.update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

    def push(self, s,a,r,s2,d):
        self.buffer.add(s,a,r,s2,d)

import matplotlib.pyplot as plt
import time

# -----------------------------
# 6. Training Loop (按 episode)
# -----------------------------
def train(num_episodes,
          checkpoint_path='checkpoints/rainbow_5/rainbow_dqn_mario.pth'):
    """
    考虑 truncated、回头、停留、死亡罚分的训练函数。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make_env() 内部已经做了 TimeLimit(env, MAX_EPISODE_STEPS)
    env    = make_env()
    agent  = Agent(env.observation_space.shape, env.action_space.n, device)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # 尝试恢复 checkpoint
    start_ep = 1
    frame_idx = 0
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        agent.online.load_state_dict(ckpt['model'])
        agent.target.load_state_dict(ckpt['model'])
        agent.opt.load_state_dict(ckpt['optimizer'])
        frame_idx = ckpt.get('frame_idx', 0)
        start_ep  = ckpt.get('episode', 0) + 1
        print(f"Resuming from episode {start_ep}, frame {frame_idx}")

    while len(agent.buffer.buffer) < BATCH_SIZE:
        state = env.reset()
        for _ in range(N_STEP):  # 随机推 N_STEP 步也行
            act = env.action_space.sample()
            next_s, r, done, info = env.step(act)
            agent.push(state, act, r, next_s, done)
            state = next_s
            if done:
                break

    reward_history     = []
    env_reward_history = []
    stage_history      = []
    status_history     = []
    durations          = []
    chunk_start        = time.time()

    for ep in range(start_ep, num_episodes + 1):
        obs           = env.reset()
        ep_reward     = 0.0
        ep_env_reward = 0.0
        prev_x        = None
        prev_life     = None

        while True:
            frame_idx += 1
            agent.frame_idx = frame_idx

            action = agent.act(obs)
            next_obs, env_r, done, info = env.step(action)

            # 检查是否 truncated
            truncated = info.get('TimeLimit.truncated', False)
            # 真正的 terminal
            done_flag = done and not truncated

            # 自定义 reward
            custom_r = env_r

            # 回头 / 停留 penalty
            x_pos = info.get('x_pos')
            if x_pos is not None:
                if prev_x is None:
                    prev_x = x_pos
                dx = x_pos - prev_x
                if dx < 0:
                    custom_r += BACKWARD_PENALTY
                elif dx == 0:
                    custom_r += STAY_PENALTY
                prev_x = x_pos

            # 死亡 penalty：life 减少
            life = info.get('life')
            if prev_life is None:
                prev_life = life
            elif life < prev_life:
                custom_r += DEATH_PENALTY
            prev_life = life

            # 存 buffer／learn，用 done_flag
            agent.push(obs, action, custom_r, next_obs, done_flag)
            agent.learn()

            obs = next_obs
            ep_reward     += custom_r
            ep_env_reward += env_r

            if done:
                break

        # 本集记录
        reward_history.append(ep_reward)
        env_reward_history.append(ep_env_reward)
        stage_history.append(env.unwrapped._stage)

        status = "TRUNCATED" if truncated else "TERMINAL"
        status_history.append(status)
        # print(f"[Episode {ep:5d}] , "
        #       f"Reward: {ep_reward:6.2f}  EnvR: {ep_env_reward:6.2f}  "
        #       f"Stage: {env.unwrapped._stage}  Status: {status}")
        
        # 每 100 集：统计、保存、画图、记录耗时
        if ep % 100 == 0:
            chunk_end = time.time()
            dur = chunk_end - chunk_start
            durations.append(dur)
            chunk_start = time.time()

            # 窗口平均
            w_env    = env_reward_history[-100:]
            w_cust   = reward_history[-100:]
            w_stage  = stage_history[-100:]
            avg_env  = np.mean(w_env)
            avg_cust = np.mean(w_cust)
            avg_stg  = np.mean(w_stage)

            print(f"[Batch {ep//100:3d} | Ep {ep:5d}] "
                  f"AvgEnvR: {avg_env:6.2f}  AvgCustR: {avg_cust:6.2f}  "
                  f"AvgStg: {avg_stg:4.1f}  Frames: {frame_idx}  "
                  f"Time(100eps): {dur/60:.2f} min"
                  f" Truncated number: {status_history[-100:].count('TRUNCATED')}")

            torch.save({
                'model':     agent.online.state_dict(),
                'optimizer': agent.opt.state_dict(),
                'frame_idx': frame_idx,
                'episode':   ep
            }, checkpoint_path)

            # 画对比图
            chunks = len(reward_history) // 100
            xs     = [i*100 for i in range(1, chunks+1)]
            avg_envs  = [np.mean(env_reward_history[(i-1)*100:i*100]) for i in range(1, chunks+1)]
            avg_custs = [np.mean(reward_history[(i-1)*100:i*100])          for i in range(1, chunks+1)]

            plt.figure(figsize=(8,4))
            plt.plot(xs, avg_envs,  marker='o', label='Env Reward')
            plt.plot(xs, avg_custs, marker='x', label='Custom Reward')
            plt.xlabel('Episodes')
            plt.ylabel('Avg Reward per 100 eps')
            plt.title('Reward Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(
                os.path.dirname(checkpoint_path),
                'reward_comparison.png'))
            plt.close()

            # 5) 新增：10 次 inference 评估
            eval_env = make_env()
            eval_rewards = []
            for _ in range(10):
                e_obs = eval_env.reset()
                done = False
                total = 0.0
                step = 0
                while not done and step < 2000:
                    a = agent.act(e_obs)
                    e_obs, r, done, _ = eval_env.step(a)
                    total += r
                    step += 1
                eval_rewards.append(total)
            eval_env.close()
            print(f"    → Eval avg over 10 eps: {np.mean(eval_rewards):.2f}")

    print("Training complete.")
    return reward_history, env_reward_history, stage_history, durations
    
if __name__ == "__main__":
    train(num_episodes=100000)
