import os
import sys
# .. 的数量取决于文件所在的文件层级
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.RL.utility.reply_queue import *

t1 = Transition(
    torch.tensor([[1, 2, 3]]),
    torch.tensor([[1, 2]]),
    torch.tensor([[4, 5, 6]]),
    torch.tensor([[1]]),
    torch.tensor([[True]])
)

t2 = Transition(
    torch.tensor([[4, 5, 6]]),
    torch.tensor([[3, 4]]),
    torch.tensor([[7, 8, 9]]),
    torch.tensor([[2]]),
    torch.tensor([[False]])
)

def test_reply_queue():
    '''
    保证批量随机采样得到的 Transition 维度正确为 B x O
    '''
    
    rq = ReplyQueue(3)
    rq.append(t1)
    rq.append(t2)
    rq.append(t1)

    sample = rq.sample(2)

    assert sample.state.shape == (2, 3)
    assert sample.action.shape == (2, 2)
    assert sample.next_state.shape == (2, 3)
    assert sample.reward.shape == (2, 1)
    assert sample.done.shape == (2, 1)

    pass
