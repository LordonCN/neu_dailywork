`"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        # 设置变量
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error") # TD_error
        # 设置 两层网络 输入当前状态 动作 以及critis 计算得到的td-error (时间差)
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s, # inpupt state
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),                  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,                                                # output units
                activation=tf.nn.softmax,                                       # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),        # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )
        # Actor 损失函数  相比PG网络将vt改成了td_error
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)               # advantage (TD_error) guided loss
        # 将损失反向传播 （这里计算寻找最大偏差 类似于惊讶程度） 使用adam优化器
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)    # minimize(-exp_v) = maximize(exp_v)
    # 调用全连接层将reward归一化为概率值  probs为一：0-1的概率列表
    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})                      # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())     # return a int
    # 所有的状态按行排列 状态 动作 tderror 三个属性放到字典里（用来查阅学习）
    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}                   # put para in dict
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)        # train
        return exp_v


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        # 这里不用action
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_ next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s, # state
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            # 通过l1层计算当前状态得到状态值V
            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),        # weights
                bias_initializer=tf.constant_initializer(0.1),                  # biases
                name='V'
            )
        # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)                               # 取平方

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):                                              # 当前状态 价值 下一状态观测值    TD_error = (r+gamma*V_next) - V_eval
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]                                 # 横过来          （当前奖励+gamma×下一步状态值）-状态估计值

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error                                                      # 通过对value值判断下一步动作是否需要调整运动幅度


sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     #  LR_C>LR_A  we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break

        