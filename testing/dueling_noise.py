# test_mario_agent.py
"""
Standalone evaluation script for the trained Mario Dueling DQN agent.
Runs a fixed number of episodes and reports per-episode and average rewards.
Usage:
    python test_mario_agent.py [num_episodes]
Default num_episodes = 10
"""
import sys
from pathlib import Path

from modules.env import make_env
from modules.DQN import DuelingDQN
from modules.replay_buffer import ReplayBuffer

import torch
import time

def test(num_episodes=10, checkpoint_path='checkpoints/mario_dueling_latest.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment and agent
    env = make_env()
    action_dim = env.action_space.n
    agent = DuelingDQN(action_dim).to(device)
    # Load policy network weights only
    cp = Path(checkpoint_path)
    if not cp.exists():
        print(f"Checkpoint not found at {cp}")
        return
    data = torch.load(cp, map_location=device)
    agent.load_state_dict(data['policy'])
    agent.eval()

    total_reward = 0.0
    for ep in range(1, num_episodes+1):
        obs = env.reset()[0]
        # Build initial 8-frame stack
        from modules.env import preprocess
        from collections import deque
        frame = preprocess(obs)
        frame_deque = deque([frame]*8, maxlen=8)

        done = False
        ep_reward = 0.0
        while not done:
            # Stack and predict
            state = torch.stack(list(frame_deque)).unsqueeze(0).to(device).float()
            with torch.no_grad():
                q = agent(state)
                action = q.argmax(1).item()

            next_obs, reward, done, info = env.step(action)
            frame_deque.append(preprocess(next_obs))
            ep_reward += reward
            env.render()
            time.sleep(0.1)
        print(f"Test Episode {ep}/{num_episodes} reward: {ep_reward}")
        total_reward += ep_reward

    avg_reward = total_reward / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    test(n)
