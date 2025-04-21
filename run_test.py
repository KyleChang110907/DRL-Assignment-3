import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from student_agent_prev import Agent
import time

# 创建环境（不需要额外的预处理 wrapper）
env   = gym_super_mario_bros.make('SuperMarioBros-v0')
env   = JoypadSpace(env, COMPLEX_MOVEMENT)
agent = Agent()

rewards = []
for i in range(10):
    obs = env.reset()  # 原始 RGB frame, shape=(240,256,3)
    done = False
    step = 0
    total_reward = 0
    prev_x = 0
    while not done and step < 2000:
        step += 1
        action = agent.act(obs)       # 直接传原始 obs
        obs, reward, done, info = env.step(action)
        total_reward += reward
        x_pos = info.get('x_pos')
        # print(f'dx:{x_pos - prev_x}, action: {action}, reward: {reward}')
        # env.render()
        # time.sleep(0.02)  # 控制渲染速度
    rewards.append(total_reward)
    # print(f"\nEpisode finished! Total reward: {total_reward}")
    # env.close()

    print(f"Episode {i+1} finished! Total reward: {total_reward}")

print(f"Average reward over 10 episodes: {sum(rewards) / len(rewards)}")