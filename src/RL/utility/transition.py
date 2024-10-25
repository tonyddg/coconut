import torch
import numpy as np

from typing import SupportsFloat
from dataclasses import dataclass

@dataclass
class Transition:
    '''
    Transition  
    应保证元素的第一维为批次
    '''
    state: torch.Tensor 
    action: torch.Tensor
    next_state: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor

    def __eq__(self, obj):
        if not isinstance(obj, Transition):
            raise Exception("Can not compare with other type")
        for key in self.__dict__.keys():
            if not (self.__dict__[key] == obj.__dict__[key]).all():
                return False
        return True

    def batch_size(self):
        return self.reward.shape[0]

def random_transition(state_dim: int, action_dim: int, batch_size: int = 1):
    '''
    生成随机 Transition
    '''
    return Transition(
        torch.rand((batch_size, state_dim)),
        torch.randint(0, 2, (batch_size, action_dim)),
        torch.rand((batch_size, state_dim)),
        torch.rand((batch_size, 1)),
        torch.rand((batch_size, 1))
    )

def make_transition_from_numpy(state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: SupportsFloat | np.ndarray, done: bool | np.ndarray):
    '''
    通过 Numpy 数组创建 Transition, 要求 state, action, next_state, reward (多 batch) 都必须是转移所有权的 Numpy 数组  
    
    根据 reward 判断是否传入多 Batch
    * reward 为单个数字 (float 或 numpy) 时为非 batch, 将自动升维
    * reward 为多元素 numpy 数组或具有维度的单元素时, 不会自动升维

    注意
    * action 可以是单个数值, 也可以是 numpy 数组
    * state 必须是 numpy 数组
    '''
    if not isinstance(reward, np.ndarray) or reward.shape == () :
        # 当 action 为单个整数时的处理
        if action.shape == ():
            action_tensor = torch.tensor(action).view(1, -1)
        else:
            action_tensor = torch.from_numpy(action).view(1, -1)

        return Transition(
            torch.from_numpy(state).view(1, -1),
            action_tensor,
            torch.from_numpy(next_state).view(1, -1),
            torch.tensor(reward).view(1, -1),
            torch.tensor(done, dtype = torch.int64).view(1, -1)
        )
    else:
        return Transition(
            torch.from_numpy(state),
            torch.from_numpy(action),
            torch.from_numpy(next_state),
            torch.from_numpy(reward),
            torch.tensor(done, dtype = torch.int64)
        )

def pack_transition_batch(pack: list[Transition]) -> Transition:
    '''
    将多条单个的 Transition 合并为一个批次
    '''
    batch_size = len(pack)
    
    pack_state = torch.zeros((batch_size,) + pack[0].state.shape[1:])
    pack_action = torch.zeros((batch_size,) + pack[0].action.shape[1:], dtype = torch.int64)
    pack_next_state = torch.zeros((batch_size,) + pack[0].next_state.shape[1:])
    pack_reward = torch.zeros((batch_size,) + pack[0].reward.shape[1:])
    pack_done = torch.zeros((batch_size,) + pack[0].reward.shape[1:], dtype = torch.int8)

    for i, iter in enumerate(pack):
        pack_state[i] = iter.state[0]
        pack_action[i] = iter.action[0]
        pack_next_state[i] = iter.next_state[0]
        pack_reward[i] = iter.reward[0]
        pack_done[i] = iter.done[0]

    return Transition(pack_state, pack_action, pack_next_state, pack_reward, pack_done)
