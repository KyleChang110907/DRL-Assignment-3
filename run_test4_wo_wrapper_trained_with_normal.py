# inference_agent.py

import torch
import numpy as np
from collections import deque
from torchvision import transforms as T
from PIL import Image
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import time

# 请确保下面这两个类已经和训练时完全一样地定义过，或者从你的训练代码里 import 进来
# from student_agent_wo_wrapper_trained_with_normal_self import InferenceAgent
if __name__ == "__main__":
    MAX_EPISODE_STEPS = 3000
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym.wrappers import TimeLimit
    
    # 1. 创建“只能这么长” 的测试环境
    env   = gym_super_mario_bros.make('SuperMarioBros-v0')
    env   = JoypadSpace(env, COMPLEX_MOVEMENT)
    env   = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

    
    # 2. 实例化推理 Agent（载入你训练好的 checkpoint）
    # from student_agent_wo_wrapper_trained_with_normal_self import InferenceAgent
    # agent = InferenceAgent('checkpoints/rainbow_11/rainbow_dqn_mario.pth', device='cpu')

    # from student_agent_wo_wrapper_trained_with_normal_self_ICM import InferenceAgent
    # agent = InferenceAgent('checkpoints/rainbow_icm/rainbow_icm.pth', device='cpu')

    # 3. 跑十集
    rewards = []
    for ep in range(10):

        # ICM
        # from student_agent_wo_wrapper_trained_with_normal_self_ICM import InferenceAgent
        # agent = InferenceAgent('checkpoints/rainbow_icm/rainbow_icm.pth', device='cpu')
        
        # Normal
        from student_agent_wo_wrapper_trained_with_normal_self import InferenceAgent
        agent = InferenceAgent('checkpoints/rainbow_11/rainbow_dqn_mario.pth', device='cpu')

        obs   = env.reset()   # raw RGB frame, shape=(240,256,3)
        done  = False
        total = 0
        steps = 0

        # 注意：第一次调用 act() 时内部队列会自动填满；不用额外清零
        while not done and steps < 3000:
            # 只用原图调用 act，内部自动做所有 wrapper 操作
            a, steps = agent.act(obs), steps + 1

            obs, r, done, info = env.step(a)
            total += r
            # print(f"Step {steps:4d}  Action {a:2d}  Reward {r:.2f}  Total {total:.2f}")
            env.render()
            time.sleep(0.02)
        print(f"Episode {ep+1:2d} → reward = {total:6.1f}")
        rewards.append(total)

    print(f"Average reward: {np.mean(rewards):6.1f}")