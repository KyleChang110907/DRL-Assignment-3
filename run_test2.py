from training.rainbow import make_env
from student_agent_w_wrapper import Agent
import time 

env = make_env()
agent = Agent()

rewards = []

for _ in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    while not done:
        step += 1
        action = agent.act(obs)
        obs, r, done, info = env.step(action)
        total_reward += r
        print(f"Step:{step}, Action: {action}, Reward: {r}, Done: {done}")
        env.render()  # Uncomment to visualize the environment
        time.sleep(0.1)  # Uncomment to control rendering speed
    print("Total reward:", total_reward)
    rewards.append(total_reward)

print("Average reward over 10 episodes:", sum(rewards) / len(rewards))
env.close()
