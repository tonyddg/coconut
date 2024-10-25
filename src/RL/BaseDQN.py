import torch
from torch import nn
from torch import optim

import numpy as np
import math

from dataclasses import dataclass

from .utility.reply_queue import *
from .utility.train_rl import RLModel

# 环境 Cart Pole
# Acion: 离散值 {0, 1}
# State: np 数组 (4,)
ACTION_NUM = 2
STATE_DIM = 4

@dataclass
class HyperParam:
    ## 基础
    # 学习率
    alpha: float = 1e-4
    # 回报折扣率
    gamma: float = 0.99

    ## epsilon-greedy
    # 随机决策概率
    epsilon_start: float = 0.90
    epsilon_final: float = 0.05
    action_take_time_decay: int = 1000
    
    ## 网络
    # 隐藏层大小
    hidden1_dim: int = 128
    # 隐藏层大小
    hidden2_dim: int = 128
    # 状态参数数
    state_dim: int = STATE_DIM
    # 可执行操作数
    action_num: int = ACTION_NUM

    ## 经验回放
    # 批次大小
    batch_size: int = 128
    # 回放队列容量
    reply_size: int = 10000
    # 开始训练所需回放
    minimal_size: int = 500

class QNet(nn.Module):
    def __init__(self, hidden1_dim: int, hidden2_dim: int, state_dim: int = STATE_DIM, action_dim: int = ACTION_NUM) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(state_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, action_dim)
        )

    def forward(self, X):
        return self.dense(X)

class BaseDQN(RLModel):

    # 初始化部分

    def __init__(self, hparams: HyperParam) -> None:
        
        self.hparams = hparams

        self.epsilon = self.hparams.epsilon_start
        self.reply_queue = ReplyQueue(self.hparams.reply_size)

        self.action_take_times = 0

        self.q_network = QNet(
            self.hparams.hidden1_dim,
            self.hparams.hidden2_dim,
            self.hparams.state_dim,
            self.hparams.action_num
        )

        # Target Network 保持在评估模式
        self.target_network = QNet(
            self.hparams.hidden1_dim,
            self.hparams.hidden2_dim,
            self.hparams.state_dim,
            self.hparams.action_num
        )
        self.target_network.eval()
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.AdamW( # type: ignore
            self.q_network.parameters(),
            self.hparams.alpha,
            amsgrad = True
        )
        # 使用均值 MSE 作为损失函数
        self.loss_fn = nn.MSELoss()

    # 决策部分

    def  _update_epsilon(self, action_take_times: int):
        '''
        epsilon-greedy 决策中, 使用指数规律更新 epsilon
        '''
        self.epsilon = \
            self.hparams.epsilon_final + (self.hparams.epsilon_start - self.hparams.epsilon_final) \
            * math.exp(-1.0 * action_take_times / self.hparams.action_take_time_decay) # 指数规律 epsilon

    def _take_random_action(self, state: torch.Tensor) -> torch.Tensor:
        # 取 0 到 action_num 的随机数
        return torch.randint(0, self.hparams.action_num, (state.size()[0], 1))

    def _take_valuable_action(self, state: torch.Tensor) -> torch.Tensor:
        # 取第一维的 Argmax (非 Batch), 但保持维度 (维度长度变为 1)
        return torch.argmax(self.q_network(state), 1, True)

    def take_action(self, state: np.ndarray):
        '''
        单次 epsilon-greedy 决策  
        传入 Numpy 数组, 同时传出 Numpy 数组
        '''
        state_tensor = torch.tensor(state).view(1, -1)

        with torch.no_grad():
            if random.random() > self.epsilon:
                action_tensor =  self._take_valuable_action(state_tensor)
            else:
                action_tensor =  self._take_random_action(state_tensor)

        self.action_take_times += 1
        self._update_epsilon(self.action_take_times)

        return action_tensor[0].numpy()[0]

    # 训练部分

    def _get_td_target(self, transition: Transition) -> torch.Tensor:
        '''
        获取 TD Target, 不会计算梯度
        '''
        with torch.no_grad():
            # 使用 DQN 预测动作
            valuable_action = self._take_valuable_action(transition.next_state)

            # 使用 Target Network 预测价值
            target_output = self.target_network(transition.next_state)
            
            # 取 Target Network 中 DQN 的预测动作作为 Q* 的预测
            mix_predict = torch.gather(target_output, 1, valuable_action)

            return transition.reward + self.hparams.gamma * mix_predict * (1.0 - transition.done)

    def _get_predict(self, transition: Transition) -> torch.Tensor:
        '''
        获取模型预测
        '''
        predict_batch = self.q_network(transition.state)
        return torch.gather(predict_batch, 1, transition.action)

    def _batch_update(self, transition: Transition) -> float:
        '''
        进行一个批次的训练
        '''

        self.q_network.train()

        # 计算 TD 目标与模型预测
        td_target = self._get_td_target(transition)
        predict = self._get_predict(transition)

        # 梯度下降更新模型
        loss = self.loss_fn(predict, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        # 裁剪梯度
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        self.q_network.eval()

        return loss.item()

    def update_step(self, transition: Transition):
        '''
        基于经验回放更新模型

        * 成功更新时返回 loss
        * 失败时返回 None
        '''

        self.reply_queue.append(transition)

        if self.reply_queue.size() < self.hparams.minimal_size:
            return None
        else:
            batch_transition = self.reply_queue.sample(self.hparams.batch_size)
            return self._batch_update(batch_transition)

    def _sync_target_network(self, net: nn.Module, target: nn.Module):
        '''
        使用直接更新法, 更新 Target Network 参数
        '''
        target.load_state_dict(net.state_dict())
        # for param_target, param in zip(target.parameters(), net.parameters()):
        #     param_target.data.copy_(param_target.data * self.hparams.tau + param.data * (1.0 - self.hparams.tau))

    def update_episode(self, episode: int):
        self._sync_target_network(self.q_network, self.target_network)

    ### 

    def take_action_deploy(self, state: np.ndarray):
        state_tensor = torch.tensor(state).view(1, -1)

        with torch.no_grad():
                action_tensor =  self._take_valuable_action(state_tensor)

        return action_tensor[0].numpy()[0]
