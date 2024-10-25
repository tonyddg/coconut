# torch.distributions.Categorical 根据概率质量分布采样
# torch.Tensor.detach() 分离出一个视为常量的张量, 没有梯度 (最好仅在最后时刻使用)
# 先按算法写一个 demo, 再在 demo 的基础上重构

import torch 
from torch import nn
from torch import optim

from dataclasses import dataclass
import numpy as np
from collections import deque

from .utility.transition import *
from .utility.train_rl import RLModel

@dataclass
class HyperParam:
    ### 网络规模
    action_num: int = 2
    state_dim: int = 4
    hidden_dim: int = 128

    ### 训练参数
    # 价值网络学习率
    lr_critic: float = 1e-2
    # 策略网络学习率
    lr_actor: float = 1e-3
    # 回报折扣
    gamma: float = 0.99

    ### 多步 TD Target (基础 A2C 此参数无效)
    # 使用 m 步的奖励用于计算 TD 目标
    m: int = 5

class PolicyNet(nn.Module):
    def __init__(self, action_num: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_num),
            nn.Softmax(1) # 输出概率质量分布
        )
    def forward(self, X):
        return self.dense(X)

class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, X):
        return self.dense(X)

class A2C(RLModel):
    def __init__(self, hyper_param: HyperParam) -> None:
        self.hyper_param = hyper_param

        self.actor = PolicyNet(
            self.hyper_param.action_num,
            self.hyper_param.state_dim,
            self.hyper_param.hidden_dim
        )
        self.actor_optimizer = optim.Adam( # type: ignore
            self.actor.parameters(),
            lr = self.hyper_param.lr_actor
        )
        
        self.critic = ValueNet(
            self.hyper_param.state_dim,
            self.hyper_param.hidden_dim
        )
        self.critic_optimizer = optim.Adam( # type: ignore
            self.critic.parameters(),
            lr = self.hyper_param.lr_critic
        )
        self.critic_loss = nn.MSELoss()

    ###

    def take_action(self, state: np.ndarray) -> np.int64:
        state_tensor = torch.from_numpy(state).view(1, -1)
        with torch.no_grad():
            action_ditribute = self.actor(state_tensor)
        
        action_sample = torch.distributions.Categorical(action_ditribute).sample()
        return action_sample.numpy()[0]

    ###

    def _get_td_target(self, v_t: torch.Tensor, transition: Transition):
        with torch.no_grad():
            v_t1 = self.critic(transition.next_state)
            y_t = transition.reward + self.hyper_param.gamma * v_t1 * (1 - transition.done)
            delta_t = y_t - v_t

            return (y_t, delta_t)

    def _optim_critic(self, v_t: torch.Tensor, y_t: torch.Tensor):
        critic_loss = self.critic_loss(v_t, y_t)
        self.critic_optimizer.zero_grad()

        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _optim_actor(self, state: torch.Tensor, action: torch.Tensor, delta_t: torch.Tensor):
        p = torch.gather(self.actor(state), 1, action)        
        actor_loss = -1 * torch.mean(torch.log(p) * delta_t.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_step(self, transition: Transition):
        v_t = self.critic(transition.state)

        y_t, delta_t = self._get_td_target(v_t, transition)
        critic_loss = self._optim_critic(v_t, y_t)
        self._optim_actor(transition.state, transition.action, delta_t)

        return critic_loss

    def update_episode(self, episode: int):
        pass

    ###

    def take_action_deploy(self, state: np.ndarray) -> np.int64:
        return self.take_action(state)

class A2C_WithMultiStep(A2C):
    def __init__(self, hyper_param: HyperParam):
        super().__init__(hyper_param)

        # 缓存队列, 保存二元组 (transition, step)
        self.cache_queue: list[Transition] = []

    def _update_cache_queue(self, transition: Transition):
        '''
        更新缓存奖励队列中的奖励, 并插入新的 Transition
        '''
        size = len(self.cache_queue)
        for i, iter in enumerate(self.cache_queue):
            iter.reward = iter.reward + transition.reward * self.hyper_param.gamma ** (size - i)
        self.cache_queue.append(transition)

    def _override_next_state(self):
        '''
        统一 next_state
        '''
        last_next_state = self.cache_queue[-1].next_state
        for iter in self.cache_queue[:-1]:
            iter.next_state = last_next_state

    def _get_td_target(self, v_t: torch.Tensor, transition: Transition):
        '''
        重载为带有 gamma ^ m 次方的多步 td
        '''
        with torch.no_grad():
            v_t1 = self.critic(transition.next_state)

            pow_sequene = torch.arange(transition.batch_size(), 0, -1).view(-1, 1)
            y_t = transition.reward + (self.hyper_param.gamma ** pow_sequene) * v_t1 * (1 - transition.done)
            delta_t = y_t - v_t

            return (y_t, delta_t)

    def update_step(self, transition: Transition):

        self._update_cache_queue(transition)
        
        loss = None
        # 每次将前 1 到 m 步的 Transition 合并为一个 batch 用于训练
        if len(self.cache_queue) == self.hyper_param.m or transition.done:
            self._override_next_state()
            batch_transition = pack_transition_batch(self.cache_queue)

            loss = super().update_step(batch_transition)
            self.cache_queue = []

        return loss
