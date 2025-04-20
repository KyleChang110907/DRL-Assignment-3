# training/dueling_noise.py
import os, random, datetime, math
from pathlib import Path
from collections import deque

import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import torch
from torch import nn
import torch.nn.functional as F

# 导入自定义环境与预处理
from modules.env import make_env, preprocess
from modules.DQN import DuelingDQN
from modules.replay_buffer import ReplayBuffer

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

action_space = COMPLEX_MOVEMENT
actions = len(action_space)

# ---------- Agent ----------
class MarioAgent:
    def __init__(self, action_dim):
        self.policy_net  = DuelingDQN(action_dim).to(DEVICE)
        self.target_net  = DuelingDQN(action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.buffer = ReplayBuffer()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.batch_size = 32
        self.update_freq = 4
        self.sync_every  = 10000  # steps
        self.step_count = 0
        self.eps_start, self.eps_end, self.eps_decay = 1.0, 0.1, 10000000
        self.frame_deque = None
        self.last_pre_stack = None
        self.last_state = None

    def reset(self, obs):
        # initialize frame buffer from raw observation
        frame = preprocess(obs)
        self.frame_deque = deque([frame] * 8, maxlen=8)

    def act(self, obs):
        # build stacks: pre-step and post-step
        frame = preprocess(obs)
        pre_stack = torch.stack(list(self.frame_deque))  # (8,84,84)
        self.last_pre_stack = pre_stack
        self.frame_deque.append(frame)
        next_stack = torch.stack(list(self.frame_deque))  # (8,84,84)
        self.last_state = next_stack
        state_input = next_stack.unsqueeze(0).to(DEVICE).float()  # (1,8,84,84)

        # exploration
        self.policy_net.eval()
        self.policy_net.reset_noise()
        if random.random() < self.current_epsilon():
            return random.randrange(actions)
        with torch.no_grad():
            q_values = self.policy_net(state_input)
        return q_values.argmax(1).item()

    def current_epsilon(self):
        # return current epsilon value
        return max(self.eps_end, self.eps_start - self.step_count / self.eps_decay)

    def learn(self):
        if len(self.buffer) < 50000 or self.step_count % self.update_freq != 0:
            return
        batch = self.buffer.sample(self.batch_size)
        states      = torch.stack(batch.state).to(DEVICE).float()
        actions_t   = torch.tensor(batch.action, device=DEVICE).unsqueeze(1)
        rewards     = torch.tensor(batch.reward, device=DEVICE).unsqueeze(1)
        next_states = torch.stack(batch.next_state).to(DEVICE).float()
        dones       = torch.tensor(batch.done, device=DEVICE).unsqueeze(1).float()

        self.policy_net.train()
        self.policy_net.reset_noise()
        q_values = self.policy_net(states).gather(1, actions_t)

        with torch.no_grad():
            self.target_net.reset_noise()
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.step_count % self.sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            'policy': self.policy_net.cpu().state_dict(),
            'target': self.target_net.cpu().state_dict(),
            'steps' : self.step_count
        }, path)
        self.policy_net.to(DEVICE)
        self.target_net.to(DEVICE)

    def load(self, path):
        data = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(data['policy'])
        self.target_net.load_state_dict(data['target'])
        self.policy_net.to(DEVICE)
        self.target_net.to(DEVICE)
        self.step_count = data.get('steps', 0)
        print(f"Resumed training from {path}, starting at step {self.step_count}")

# ---------- Main training loop by episodes ----------
def main(num_episodes=100000, save_dir='checkpoints'):
    env = make_env()
    agent = MarioAgent(env.action_space.n)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = Path(save_dir) / 'mario_dueling_latest.pth'
    if checkpoint_path.exists():
        agent.load(checkpoint_path)

    if DEVICE.type == "cuda":
        dummy = torch.zeros((1, 8, 84, 84), device=DEVICE)
        _ = agent.policy_net(dummy.float())
        print(f"[GPU] allocated={torch.cuda.memory_allocated()/1e6:.2f}MB | reserved={torch.cuda.max_memory_reserved()/1e6:.2f}MB")

    for episode in range(1, num_episodes+1):
        obs = env.reset()[0]
        agent.reset(obs)
        episode_reward = 0
        done = False

        while not done:
            action = agent.act(obs)
            prev_stack = agent.last_pre_stack.cpu()
            next_obs, reward, done, info = env.step(action)
            next_stack = agent.last_state.cpu()

            agent.buffer.push(prev_stack, action, reward, next_stack, done)
            agent.step_count += 1
            agent.learn()
            episode_reward += reward
            obs = next_obs

        # Print episode stats including epsilon
        eps = agent.current_epsilon()
        stage = env.unwrapped._stage
        print(f'Episode {episode}/{num_episodes} | reward {episode_reward} | steps {agent.step_count} | stage {stage} | epsilon {eps:.4f}')
        agent.save(checkpoint_path)

    print(f'Finished {num_episodes} episodes, latest model at {checkpoint_path}')
    env.close()

if __name__ == "__main__":
    main()
