import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#####算法流程####
# 输入：迭代轮数T，状态集S，动作集A，步长alpha，衰减gama，探索率episil
# 输出：所有状态和动作对应的价值Q

# 算法步骤：
# 1	随机初始化所有状态和动作对应的价值/收获Q，对于终止状态的Q值初始化为0；
# 2	迭代开始————1：T，进行迭代
# 3		初始化S为当前状态序列的第一个状态;
# 4		设置A为episil贪婪法在当前状态S选择的动作，在状态S执行动作A，得到新状态S’和奖励R;
# 5		用episil贪婪法在状态S'选择新的动作A'；
# 6		更新价值函数Q(S,A)=Q(S,A)+alpha*(R+gama*Q(S',A')-Q(S,A));
# 7		S = S'; A =A’;
# 8		如果S'为终止状态，当前轮迭代完毕，否则转到步骤#4

# 状态集合S=[i,j]; i = 0:WINDYGRID_WIDTH-1; j = 0:WINDYGRID_LENGTH-1;
WINDYGRID_LENGTH = 10
WINDYGRID_WIDTH = 7
# 动作集合A
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
# Q(S,A)表初始化
q_value = np.zeros((WINDYGRID_WIDTH, WINDYGRID_LENGTH, len(ACTIONS)))

# 其他参数：迭代轮数，步长，衰减，探索率
ITER_MAX = 1000  # 最大迭代轮数
EPSILON = 0.1  # epsilon贪婪法中的探索率
ALPHA = 0.5  # 步长
REWARD = -1.0  # 奖励
GAMA = 1.0  # 衰减
START = [3, 0]
GOAL = [3, 7]
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


# 按照windyGrid的规则和选择的动作进行移动，找到S'
def step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WINDYGRID_WIDTH - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WINDYGRID_LENGTH - 1)]
    else:
        assert False


def sarsa():
    for iter in range(1, ITER_MAX):
        # 初始化当前状态为起始位置
        state = START
        # 在当前状态下采用episl贪婪法选择动作A
        if np.random.binomial(1, EPSILON) == 1:
            action = np.random.choice(ACTIONS)
        else:
            values = q_value[state[0], state[1], :]
            action = np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])

        while state != GOAL:
            next_state = step(state, action)

            # 在状态S'下采用epsilon贪婪法选择动作A'
            if np.random.binomial(1, EPSILON) == 1:
                next_action = np.random.choice(ACTIONS)
            else:
                values = q_value[next_state[0], next_state[1], :]
                next_action = np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])

            # 更新Q表
            q_value[state[0], state[1], action] += \
                ALPHA * (REWARD + GAMA * q_value[next_state[0], next_state[1], next_action] -
                         q_value[state[0], state[1], action])
            state = next_state
            action = next_action

    # 经历了ITER_MAX轮后，Q表已经收敛，下面就要将Q表映射为最优策略
    optimal_policy = []
    for i in range(0, WINDYGRID_WIDTH):
        optimal_policy.append([])
        for j in range(0, WINDYGRID_LENGTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            if bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            if bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            if bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy is :')
    for row in optimal_policy:
        print(row)
    print(WIND)


if __name__ == '__main__':
    sarsa()

# ['R', 'U', 'D', 'R', 'R', 'R', 'R', 'R', 'R', 'D']
# ['U', 'R', 'R', 'R', 'R', 'R', 'U', 'R', 'R', 'D']
# ['U', 'R', 'R', 'R', 'R', 'R', 'D', 'R', 'R', 'D']
# ['R', 'R', 'R', 'R', 'R', 'R', 'L', 'G', 'R', 'D']
# ['D', 'D', 'R', 'R', 'R', 'R', 'U', 'D', 'L', 'L']
# ['R', 'R', 'R', 'R', 'R', 'U', 'U', 'U', 'D', 'D']
# ['R', 'R', 'R', 'R', 'U', 'U', 'U', 'U', 'U', 'L']
# ['0', '0', '0', '1', '1', '1', '2', '2', '1', '0']