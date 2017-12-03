import gym
import numpy as np

env = gym.make('Pendulum-v0')

# 행동, 상태, 보상 확인

print(env.action_space)
print(env.observation_space)
print(env.reward_range)

# 행동의 개수 확인
print(env.action_space.high)
print(env.action_space.low)

for i_episode in range(2):
    observation = env.reset()
    rewards = []
    total_reward = 0
    while True:
        env.render()
        action = env.action_space.sample()
        # print(action)
        observation, reward, done, info = env.step(action)

        total_reward += reward

        # if reward not in rewards:
        #     rewards.append(reward)
        #     # print("observation : ", observation[50])
        #     print("state : ", observation)
        #     print("action : ", action)
        #     print("reward : ", reward)
        #     print("done : ", done)
        #     print("info : ", info)

        if done:
            print("Episode is done")
            print("total reward : ", total_reward)
            break

env.close()
