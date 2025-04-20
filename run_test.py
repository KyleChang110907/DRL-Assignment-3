import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from student_agent_test import Agent
import time

# 创建环境（不需要额外的预处理 wrapper）
env   = gym_super_mario_bros.make('SuperMarioBros-v0')
env   = JoypadSpace(env, COMPLEX_MOVEMENT)
agent = Agent()

obs = env.reset()  # 原始 RGB frame, shape=(240,256,3)
done = False
step = 0
total_reward = 0
while not done and step < 5000:
    step += 1
    action = agent.act(obs)       # 直接传原始 obs
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()
    time.sleep(0.02)  # 控制渲染速度
print(f"\nEpisode finished! Total reward: {total_reward}")
env.close()
