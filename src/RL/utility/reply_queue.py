from .transition import *
from collections import deque
import random

class ReplyQueue:
    def __init__(self, capacity: int) -> None:
        '''
        经验队列
        '''
        # 当 deque 已满时插入元素, 另一端元素将自动弹出
        self.buffer: deque[Transition] = deque(maxlen = capacity)

    def size(self):
        return len(self.buffer)

    def append(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        '''
        均匀采样, 当经验回放中的 Transition 不足时将抛出异常
        '''
        if batch_size > self.size():
            raise Exception(f"经验队列内的经验数 {self.size()} 小于采样数 {batch_size}")
        return pack_transition_batch(random.sample(self.buffer, batch_size))
