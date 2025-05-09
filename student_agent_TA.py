import gym
from training.dueling_noise import MarioAgent  # adjust import path as needed
import os

# Do not modify the input of the 'act' function and the '__init__' function. 
import gym

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

    def act(self, observation):
        return self.action_space.sample()