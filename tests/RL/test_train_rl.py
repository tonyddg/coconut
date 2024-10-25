import os
import sys
# .. 的数量取决于文件所在的文件层级
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.RL.utility.train_rl import *

def test_rl_teacher():
    env = gym.make(
        "CartPole-v1"
    )
    model = SampleModel(env)

    teacher = RL_Teacher(model, "test", f"test", id = "CartPole-v1")
    teacher.train(episode = 300, is_log = False)
    teacher.test(is_log_vedio = False)
