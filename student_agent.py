import gym
from training.dueling_noise import MarioAgent  # adjust import path as needed
import os

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

        # Instantiate your MarioAgent
        self.mario = MarioAgent(self.action_space.n)
        # Load the latest checkpoint
        ckpt_path = "checkpoints/mario_dueling_latest.pth"
        if os.path.isfile(ckpt_path):
            self.mario.load(ckpt_path)
        else:
            print("Warning: no checkpoint found, using random weights.")

        # We’ll feed raw obs to self.mario.act(), which expects obs
        self._inited = False

    def act(self, observation):
        """
        observation: raw RGB HxWx3 numpy array
        returns: integer action in [0,11]
        """
        # On first call, initialize the frame deque inside MarioAgent
        if not self._inited:
            self.mario.reset(observation)
            self._inited = True

        # Delegate to your MarioAgent
        action = self.mario.act(observation)
        return action

# just for testing 
