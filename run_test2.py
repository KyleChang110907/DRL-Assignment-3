from training.rainbow import make_env
from student_agent import Agent

env = make_env()
agent = Agent()

rewards = []

for _ in range(10):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(obs)
        obs, r, done, info = env.step(action)
        total_reward += r

    print("Total reward:", total_reward)
    rewards.append(total_reward)

print("Average reward over 10 episodes:", sum(rewards) / len(rewards))
env.close()
