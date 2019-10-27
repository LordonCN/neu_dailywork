"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

for i_episode in range(3000):# 3000回合

    observation = env.reset() # 当前点的观测值

    while True:
        # if RENDER: env.render()

        action = RL.choose_action(observation) # 根据当前observation选action

        observation_, reward, done, info = env.step(action)# 执行动作 返回下一步观测值 奖励 完成状态 

        RL.store_transition(observation, action, reward)# 保存当前观测值 动作 奖励

        if done:
            ep_rs_sum = sum(RL.ep_rs)# 一回合完成后，将每一步得到的奖励进行累计求和 

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum # 局部变量设一个running_reward
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()# 对这一回合中每一步的：观测值 动作 奖励 进行学习    学习3000次

            if i_episode == 0:# 执行完3000之后
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_ # 如未完成 继续执行
