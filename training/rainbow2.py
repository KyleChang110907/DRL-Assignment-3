import os
import time
from collections import deque, namedtuple

import cv2
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------------
# Hyperparameters
# -----------------------------
COPY_NETWORK_FREQ       = 10000
BUFFER_CAPACITY         = 10000
BATCH_SIZE              = 32
GAMMA                   = 0.9
LEARNING_RATE           = 0.00025
ADAM_EPS                = 0.00015
PER_ALPHA               = 0.6
PER_BETA_START          = 0.4
PER_BETA_FRAMES         = 2_000_000
N_STEP                  = 5
NOISY_SIGMA_INIT        = 2.5
BACKWARD_PENALTY        = 0
STAY_PENALTY            = 0
DEATH_PENALTY           = -100
MAX_EPISODE_STEPS       = 2000

# -----------------------------
# Noisy Linear Layer
# -----------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma_init=NOISY_SIGMA_INIT):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.weight_mu    = nn.Parameter(torch.empty(out_f, in_f))
        self.weight_sigma = nn.Parameter(torch.empty(out_f, in_f))
        self.register_buffer('weight_epsilon', torch.empty(out_f, in_f))
        self.bias_mu      = nn.Parameter(torch.empty(out_f))
        self.bias_sigma   = nn.Parameter(torch.empty(out_f))
        self.register_buffer('bias_epsilon', torch.empty(out_f))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / (self.in_f ** 0.5)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init/(self.in_f ** 0.5))
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_sigma, self.sigma_init/(self.out_f ** 0.5))

    def reset_noise(self):
        def f(x): return x.sign() * x.abs().sqrt()
        eps_in = f(torch.randn(self.in_f))
        eps_out = f(torch.randn(self.out_f))
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

# -----------------------------
# Dueling CNN
# -----------------------------
class DuelingCNN(nn.Module):
    def __init__(self, in_c, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),   nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),   nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_c, 84, 90)
            n_flat = self.features(dummy).shape[1]
        self.val_noisy = NoisyLinear(n_flat, 512)
        self.val       = NoisyLinear(512, 1)
        self.adv_noisy = NoisyLinear(n_flat, 512)
        self.adv       = NoisyLinear(512, n_actions)

    def forward(self, x):
        x = self.features(x / 255.0)
        v = F.relu(self.val_noisy(x)); v = self.val(v)
        a = F.relu(self.adv_noisy(x)); a = self.adv(a)
        return v + (a - a.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for m in [self.val_noisy, self.val, self.adv_noisy, self.adv]:
            m.reset_noise()

# -----------------------------
# PER + N‑step Replay Buffer
# -----------------------------
class PrioritizedReplayBuffer:
    def __init__(self, cap, alpha, beta_start, beta_frames, n_step, gamma):
        self.cap = cap; self.alpha = alpha
        self.beta_start = beta_start; self.beta_frames = beta_frames
        self.beta_by_frame = lambda f: min(1.0,
            beta_start + f*(1.0-beta_start)/beta_frames)
        self.n_step = n_step; self.gamma = gamma
        self.buffer = []; self.prios = np.zeros((cap,), dtype=np.float32)
        self.pos = 0; self.n_buf = deque(maxlen=n_step)
        self.Exp = namedtuple('Exp', ['s','a','r','s2','d'])

    def _get_n_step(self):
        r, s2, d = self.n_buf[-1].r, self.n_buf[-1].s2, self.n_buf[-1].d
        for trans in reversed(list(self.n_buf)[:-1]):
            r = trans.r + self.gamma * r * (1 - trans.d)
            s2, d = (trans.s2, trans.d) if trans.d else (s2, d)
        return r, s2, d

    def add(self, s, a, r, s2, d):
        self.n_buf.append(self.Exp(s,a,r,s2,d))
        if len(self.n_buf) < self.n_step: return
        r_n, s2_n, d_n = self._get_n_step()
        s0, a0 = self.n_buf[0].s, self.n_buf[0].a
        exp = self.Exp(s0, a0, r_n, s2_n, d_n)
        if len(self.buffer) < self.cap:
            self.buffer.append(exp)
            prio = 1.0 if len(self.buffer)==1 else self.prios.max()
        else:
            self.buffer[self.pos] = exp; prio = self.prios.max()
        self.prios[self.pos] = prio
        self.pos = (self.pos + 1) % self.cap

    def sample(self, bs, frame_idx):
        N = len(self.buffer)
        if N == 0: return [], [], [], [], [], [], []
        prios = self.prios[:N] ** self.alpha
        sum_p = prios.sum()
        probs = prios/sum_p if sum_p>0 else np.ones_like(prios)/N
        idxs = np.random.choice(N, bs, p=probs)
        batch = self.Exp(*zip(*[self.buffer[i] for i in idxs]))
        beta = self.beta_by_frame(frame_idx)
        weights = (N * probs[idxs]) ** (-beta)
        weights /= weights.max()
        return (np.array(batch.s), batch.a, batch.r,
                np.array(batch.s2), batch.d,
                weights.astype(np.float32), idxs)

    def update_priorities(self, idxs, errors):
        for i, e in zip(idxs, errors): self.prios[i] = abs(e) + 1e-6

# -----------------------------
# Agent
# -----------------------------
class Agent:
    def __init__(self):
        self.obs_c, self.h, self.w = 4, 84, 90
        self.n_actions = len(COMPLEX_MOVEMENT)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online = DuelingCNN(self.obs_c, self.n_actions).to(self.device)
        self.target = DuelingCNN(self.obs_c, self.n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.opt = optim.Adam(self.online.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

        self.buffer = PrioritizedReplayBuffer(
            BUFFER_CAPACITY, PER_ALPHA, PER_BETA_START,
            PER_BETA_FRAMES, N_STEP, GAMMA
        )
        self.gamma = GAMMA; self.batch_size = BATCH_SIZE
        self.frame_idx = 0; self.update_freq = COPY_NETWORK_FREQ

        self.frames = deque(maxlen=self.obs_c)
        self.last_state = None

    def _process(self, raw):
        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img = resized.astype(np.float32)
        self.frames.append(img)
        return np.stack(self.frames, axis=0)

    def act(self, raw):
        if len(self.frames) == 0:
            gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
            img = resized.astype(np.float32)
            for _ in range(self.obs_c): self.frames.append(img)
        state = self._process(raw)
        self.last_state = state
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): q = self.online(tensor)
        return int(q.argmax(1).item())

    def push(self, s,a,r,s2,d): self.buffer.add(s,a,r,s2,d)

    def learn(self):
        if self.frame_idx < self.batch_size: return
        s,a,r,s2,d,w,idxs = self.buffer.sample(self.batch_size, self.frame_idx)
        s = torch.from_numpy(s).to(self.device)
        s2= torch.from_numpy(s2).to(self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        w = torch.tensor(w, dtype=torch.float32, device=self.device)

        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        a_next = self.online(s2).argmax(1)
        q_next = self.target(s2).gather(1, a_next.unsqueeze(1)).squeeze(1)
        q_tar  = r + (self.gamma**N_STEP) * q_next * (1.0 - d)
        td     = q_pred - q_tar.detach()
        loss   = (F.smooth_l1_loss(q_pred, q_tar.detach(), reduction='none') * w).mean()

        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.online.reset_noise(); self.target.reset_noise()
        self.buffer.update_priorities(idxs, td.detach().cpu().numpy())
        if self.frame_idx % self.update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

# -----------------------------
# Training Loop
# -----------------------------
import gym

def train(num_episodes, checkpoint_path='checkpoints/rainbow_22/rainbow_mario.pth'):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

    agent = Agent()
    raw = env.reset()
    action = agent.act(raw)
    state = agent.last_state

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    start_ep, fi = 1, 0
    if os.path.isfile(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=agent.device)
        agent.online.load_state_dict(ck['model'])
        agent.target.load_state_dict(ck['model'])
        agent.opt.load_state_dict(ck['optimizer'])
        fi = ck.get('frame_idx', 0); start_ep = ck.get('episode', 0) + 1
    agent.frame_idx = fi

    # warm-up
    while len(agent.buffer.buffer) < BATCH_SIZE:
        action = agent.act(raw)
        raw_n, r, done, info = env.step(action)
        next_state = agent.last_state
        agent.push(state, action, r, next_state, done)
        raw, state = raw_n, next_state
        if done: raw, state = env.reset(), None

    history = {'reward':[], 'env_reward':[], 'stage':[], 'status':[], 'dur':[], 'avg_reward':[], 'batches':[]}
    chunk = time.time()

    for ep in range(start_ep, num_episodes+1):
        raw = env.reset(); action = agent.act(raw); state = agent.last_state
        ep_r, ep_er = 0.0, 0.0
        prev_x, prev_life = None, None
        done = False
        while not done:
            agent.frame_idx += 1
            raw_n, r_env, done, info = env.step(action)
            prev_action = action
            truncated = info.get('TimeLimit.truncated', False)
            done_flag = done and not truncated
            cr = r_env
            x_pos, life = info.get('x_pos'), info.get('life')
            if x_pos is not None:
                if prev_x is None: prev_x = x_pos
                dx = x_pos - prev_x
                cr += BACKWARD_PENALTY if dx < 0 else STAY_PENALTY if dx == 0 else 0
                prev_x = x_pos
            if prev_life is None: prev_life = life
            elif life < prev_life: 
                cr += DEATH_PENALTY
                prev_life = life
            action = agent.act(raw_n)
            next_state = agent.last_state
            agent.push(state, prev_action, cr, next_state, done_flag)
            agent.learn()
            raw, state = raw_n, next_state
            ep_r += cr; ep_er += r_env
            # env.render()

        history['reward'].append(ep_r)
        history['env_reward'].append(ep_er)
        history['stage'].append(env.unwrapped._stage)
        history['status'].append('TRUNCATED' if truncated else 'TERMINAL')
        # print(f"[Episode {ep}] | EnvR {ep_er:.2f} | CustR {ep_r:.2f} | Stg {env.unwrapped._stage} | Steps {agent.frame_idx} | Status {history['status'][-1]}")

        if ep % 50 == 0:
            dur = time.time() - chunk; chunk = time.time(); history['dur'].append(dur)
            ae = np.mean(history['env_reward'][-50:]); ac = np.mean(history['reward'][-50:]); ast = np.mean(history['stage'][-50:])
            print(f"[Batch {ep//50} Ep {ep}] EnvR {ae:.2f} CustR {ac:.2f} Stg {ast:.1f} Time {dur/60:.2f}min")
            history['batches'].append(ep); history['avg_reward'].append(ac)
            torch.save({'model':agent.online.state_dict(), 'optimizer':agent.opt.state_dict(), 'frame_idx':agent.frame_idx, 'episode':ep}, checkpoint_path)

            plt.figure(figsize=(8,4))
            plt.plot(history['batches'], history['avg_reward'], marker='o')
            plt.xlabel('Episodes'); plt.ylabel('Avg Reward/50eps'); plt.grid(True)
            plt.savefig(os.path.join(os.path.dirname(checkpoint_path), 'avg_reward_history.png'))
            plt.close()

            eval_rews=[]
            agent.online.eval()
            for _ in range(5):
                raw_e=env.reset(); action_e=agent.act(raw_e); done_e=False; tot=0.0
                while not done_e:
                    raw_e_n,r_e,done_e,_=env.step(action_e); action_e=agent.act(raw_e_n); tot+=r_e
                eval_rews.append(tot)
            print(f"    -> Eval 5 eps: mean {np.mean(eval_rews):.2f}, std {np.std(eval_rews):.2f}")
            agent.online.train()

    print("Training complete.")
    plt.figure(figsize=(8,4)); plt.plot(history['batches'],history['avg_reward'],marker='o'); plt.xlabel('Episodes'); plt.ylabel('Avg Reward/50eps'); plt.grid(True); plt.show()
    return history

if __name__ == '__main__':
    train(num_episodes=100000)
