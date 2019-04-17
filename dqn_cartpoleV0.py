##############################################################################################################
# 环境模型：OpenAIGym里的env模拟，可以获取某动作下的状态
# 算法核心：用Q网络来近似得到价值函数，关键核心是从经验放回池中随机获取状态序列后的训练过程
# 算法步骤：初始化网络结构和训练方法，步骤一、初始化状态和动作/经历epsilon，
#                               步骤二，计算reward，如果下一动作是终点，回到步骤一，否则将该序列放入经验放回池
#                               步骤三，从经验放回池中随机采样，训练Q网络参数
##############################################################################################################

import gym
from collections import deque  # deque双向队列 https://www.cnblogs.com/zhenwei66/p/6598996.html
import tensorflow as tf
import numpy as np
import random

INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
BATCH_SIZE = 32
REPLAY_SIZE = 10000
GAMMA = 0.9


class DQN():
    # DQN智能体
    def __init__(self, env):
        # 初始化经验放回集合D
        self.replay_buffer = deque()
        # 初始化参数
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]  # 状态数
        self.action_dim = env.action_space.n  # 动作数

        # 初始化网络结构及训练方法
        self.create_Q_network()
        self.creat_training_method()

        # 初始化会话
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # 网络权重:第一层：输入状态层--中间层；第二层：中间层--输出层
        # 定义网络结构及初始化参数
        W1 = self.weight_variable([self.state_dim, 20])  # 中间层节点数：20
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # 输入层占位：tf.placehoder()表示TensorFlow里的占位符，此时并没有输入数据，等session启动时，通过feed_dict传入数据
        # tf.placeholder(type,shape,name),type表示数据类型，shape表示输入维度，name表示名称
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # 隐藏层 tf.nn.relu(features,name=None),激活函数，max(features,0)，大于零的树保持不变，小于零的数置零
        # tf.matmul()矩阵相乘，tf.multiply()矩阵对应元素相乘
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        # 输出层Q值
        self.Q_value = tf.matmul(h_layer, W2) + b2

    def creat_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        # 提取经验放回池中样本对应的Q值，此时Q_value对应所有动作的Q值
        # tf.reduce_sum用于降维求和
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        # tf.reduce_mean求均值
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    def train_Q_network(self):
        self.time_step += 1
        # 步骤1：从经验放回池中随机抽取序列
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # 计算y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def weight_variable(self, shape):
        # tf.truncated_normal(shape,mean,stddev)表示从截取正态分布中取随机值，
        # shape表示输出正态分布的维度，mean表示截取正态分布的均值，stddev表示标准差
        # 在均值正负二倍标准差之外的生成值将被丢弃，使得生成值在均值附近
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        # tf.constant(value/list,shape)，生成常量表，可传入值或list，shape表示生成张量的维度
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


ENV_NAME = 'CartPole-v0'
EPISODE = 3000
STEP = 300
TEST = 10


def main():
    # 初始化OpenAI Gym环境和dqn代理
    env = gym.make(ENV_NAME)  # 初始化CartPole-V0环境
    agent = DQN(env)  # 构造DQN类，描述agent智能体

    for episode in range(EPISODE):
        # 初始化任务
        state = env.reset()
        # 训练
        for step in range(STEP):
            action = agent.egreedy_action(state)  # 根据eplison贪婪算法选出动作
            next_state, reward, done, _ = env.step(action)
            # 定义奖励
            reward = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            # 每隔100个序列进行测试
            if episode % 100 == 0:
                total_reward = 0
                for i in range(TEST):
                    state = env.reset()
                    for j in range(STEP):
                        env.render()
                        action = agent.action(state)
                        state, reward, done, _ = env.step(action)
                        total_reward += reward
                        if done:
                            break
                ave_reward = total_reward / TEST
                print('episode: ', episode, 'Evaluation Average Reward: ', ave_reward)


if __name__ == '__main__':
    main()
