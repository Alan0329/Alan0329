import torch
import numpy as np
import os


def policy(model, state, epsilon):
    # ----------------------------- epsilon greedy policy ----------------------------------------------
    # 他現在是隨機選
    action = np.random.randint(3)

    # greedy policy ，有一定機率不會選agent要選的動作，為了不要讓agent只選獎勵最大的動作
    if np.random.rand() > epsilon:
        # state=[self.position_value] + self.history #to(device)讓電腦在GPU上運算
        action = model(torch.Tensor(
            np.array(state, dtype=np.float32)).view(1, -1))
        action = np.argmax(action.data)  # 選獎勵最大的動作
    return action


def shuffle_tensor(size):
    shuffle_index = torch.randperm(size)  # 随机打乱一个数字序列。
    return shuffle_index


if __name__ == "__main__":
    shuffled_tensor = shuffle_tensor(32, 'cpu')
    print(shuffled_tensor)
