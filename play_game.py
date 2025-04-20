# play_one_game.py

import os
import gym
import time     
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# Adjust this import to wherever you defined your Agent class
from student_agent import Agent  

def play_one_game():
    # 1) Build environment
    env = gym_super_mario_bros.make(
        'SuperMarioBros-v0',
    )
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # 2) Instantiate agent
    agent = Agent()

    # 3) Reset env & agent
    obs= env.reset()[0]     # Gymnasium style
    if hasattr(agent, 'reset'):
        agent.reset(obs)     # needed if Agent wraps MarioAgent

    total_reward = 0
    done = False

    # 4) Run episode loop
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.1)  # Slow down rendering for better visualization

    print(f"\nEpisode finished! Total reward: {total_reward}")
    env.close()
    return total_reward

if __name__ == "__main__":
    # Optional: disable rendering by passing render=False
    play_one_game()
