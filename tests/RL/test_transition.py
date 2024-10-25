import os
import sys
# .. 的数量取决于文件所在的文件层级
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.RL.utility.transition import *

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

t3 = Transition(
    torch.tensor([[1, 2, 3]]),
    torch.tensor([[1]]),
    torch.tensor([[4, 5, 6]]),
    torch.tensor([[1]]),
    torch.tensor([[True]])
)


def test_pack_transition_batch():
    '''
    保证 Transition 沿 Batch 维度正确合并
    '''

    t_desire = Transition(
        torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3]]),
        torch.tensor([[1, 2], [3, 4], [1, 2]]),
        torch.tensor([[4, 5, 6], [7, 8, 9], [4, 5, 6]]),
        torch.tensor([[1], [2], [1]]),
        torch.tensor([[True], [False], [True]])
    )  

    t_test = pack_transition_batch([t1, t2, t1])
    
    assert t_desire == t_test

def test_make_transition():
    '''
    保证基于 numpy 数组的 Transition 能正确创建 
    '''

    # 一般情况
    t_desire1 = t1

    t_test1 = make_transition_from_numpy(
        np.array([1, 2, 3]),
        np.array([1, 2]),
        np.array([4, 5, 6]),
        1,
        True
    )

    assert t_desire1 == t_test1

    # 传入 Batch 的情况
    t_desire2 = pack_transition_batch([t1, t2, t1])

    t_test2 = make_transition_from_numpy(
        np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]]),
        np.array([[1, 2], [3, 4], [1, 2]]),
        np.array([[4, 5, 6], [7, 8, 9], [4, 5, 6]]),
        np.array([[1], [2], [1]]),
        np.array([[True], [False], [True]])
    )

    assert t_desire2 == t_test2

    # Action 为单个值的情况
    t_desire3 = t3

    t_test3 = make_transition_from_numpy(
        np.array([1, 2, 3]),
        np.array(1),
        np.array([4, 5, 6]),
        1,
        True
    )

    assert t_desire3 == t_test3
