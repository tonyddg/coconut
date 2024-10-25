# nn.Tanh() 保证输出为 [-1, 1], 便于使用 action_bound 映射到 action 空间
# torch.randn() 生成满足正态分布的噪声
# 更新参数时, 通过 torch.mean 将各个 batch 的值合并为单个值 (更新参数只能对标量求导)
# 如果梯度上升, 则求导变量要乘上 -1
# !!! 重要: θ′ ← τ θ + (1 −τ )θ′ ; τ = 0.005 ; 更新 Target Network 中, 原参数 θ 为小量 ; 每个 step 或 episode 更新
# 当收敛后又多次发散无法恢复 -> 降低学习率 ; 始终距离收敛存在距离, loss 没有下降 -> 增加学习率 ; loss 爆炸 -> 高估问题

# tau 越大, 高估问题越严重
# 使用 TD3 时, 可选择略大的 tau 与 lr_actor; DDPG 应使用极小的 tau 与 lr_actor 
# DDPG 与 TD3 的损失函数特点为稳定训练时, 损失函数的值应当在一个数量级内波动; 收敛时则缓慢下降

import torch
from torch import nn
from torch import optim

from dataclasses import dataclass, field

from .utility.transition import Transition
from .utility.reply_queue import *
from .utility.train_rl import RLModel

import torch.nn.functional as F

@dataclass
class HyperParam:
    ### 网络规模
    action_dim: int = 1
    state_dim: int = 3
    hidden_dim: int = 64
    action_bound: torch.Tensor = field(default_factory = lambda: torch.Tensor([2]))
    # action_bound: float = 2

    ### 训练参数
    lr_actor: float = 1e-3
    lr_critic: float = 5e-3
    gamma: float = 0.98

    ### 经验队列
    reply_queue_size: int = 10000
    batch_size: int = 64
    minimal_size: int = 1000

    ### 动作噪声
    sigma: float = 0.01

    ### 软更新加权系数
    tau: float = 0.005

    ### TD3 策略网络更新周期
    actor_update_period: int = 10

class PolicyNet(nn.Module): # problem
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, action_bound: torch.Tensor) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # 保证输出为 -1 到 1
        )
        self.action_bound = action_bound
    
    def forward(self, X: torch.Tensor):
        # 对应位置乘以 action 范围
        return self.dense(X) * self.action_bound

class ValueNet(nn.Module): # ok
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, s: torch.Tensor, a: torch.Tensor):
        # 将输入的 s 与 a 拼接
        X = torch.cat([s, a], dim = 1)
        return self.dense(X)

class DDPG(RLModel):
    def __init__(self, hyper_param: HyperParam) -> None:
        self.hyper_param = hyper_param

        ###

        self.actor = PolicyNet(
            self.hyper_param.state_dim, 
            self.hyper_param.hidden_dim, 
            self.hyper_param.action_dim,
            self.hyper_param.action_bound
        )
        self.actor_optim = optim.Adam( # type: ignore
            self.actor.parameters(),
            lr = self.hyper_param.lr_actor
        )

        self.target_actor = PolicyNet(
            self.hyper_param.state_dim, 
            self.hyper_param.hidden_dim, 
            self.hyper_param.action_dim,
            self.hyper_param.action_bound
        )
        self.target_actor.load_state_dict(self.actor.state_dict())

        ###

        self.critic = ValueNet(
            self.hyper_param.state_dim, 
            self.hyper_param.hidden_dim, 
            self.hyper_param.action_dim
        )
        self.critic_optim = optim.Adam( # type: ignore
            self.critic.parameters(),
            lr = self.hyper_param.lr_critic
        )

        self.target_critic = ValueNet(
            self.hyper_param.state_dim, 
            self.hyper_param.hidden_dim, 
            self.hyper_param.action_dim
        )
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.critic_loss_fn = nn.MSELoss()

        ### 

        self.reply_queue = ReplyQueue(self.hyper_param.reply_queue_size)

    def take_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).view(1, -1)
            predict_action: torch.Tensor = self.actor(state_tensor)

            # 使用高斯噪声
            noisy_action = predict_action + torch.randn(predict_action.size()) * self.hyper_param.sigma
            return noisy_action.numpy()[0]

    def _batch_optim(self, transition: Transition):
        with torch.no_grad():
            a_t1 = self.target_actor(transition.next_state)
            q_t1 = self.target_critic(transition.next_state, a_t1)
            y_t = transition.reward + self.hyper_param.gamma * q_t1 * (1.0 - transition.done)

        q_t = self.critic(transition.state, transition.action)
        critic_loss: torch.Tensor = self.critic_loss_fn(q_t, y_t)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()   

        a_t = self.actor(transition.state)
        actor_loss: torch.Tensor = -1 * torch.mean(self.critic(transition.state, a_t))

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._sync_target_network(self.actor, self.target_actor)
        self._sync_target_network(self.critic, self.target_critic)

        return critic_loss

    def update_step(self, transition: Transition) -> None | float:
        self.reply_queue.append(transition)

        if self.reply_queue.size() > self.hyper_param.minimal_size:
            batch_transition = self.reply_queue.sample(self.hyper_param.batch_size)
            return self._batch_optim(batch_transition).item()
        else:
            return None
    
    ###

    def _sync_target_network(self, net: nn.Module, target: nn.Module):
        '''
        使用直接更新法, 更新 Target Network 参数
        '''
        for param_target, param in zip(target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.hyper_param.tau) + param.data * self.hyper_param.tau)

    def update_episode(self, episode: int):
        pass

    ###

    def take_action_deploy(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).view(1, -1)
            predict_action: torch.Tensor = self.actor(state_tensor)
            return predict_action.numpy()[0]

class TD3(DDPG):
    def __init__(self, hyper_param: HyperParam) -> None:
        super().__init__(hyper_param)

        self.critic2 = ValueNet(
            self.hyper_param.state_dim, 
            self.hyper_param.hidden_dim, 
            self.hyper_param.action_dim
        )
        self.critic2_optim = optim.Adam( # type: ignore
            self.critic2.parameters(),
            lr = self.hyper_param.lr_critic
        )

        self.target_critic2 = ValueNet(
            self.hyper_param.state_dim, 
            self.hyper_param.hidden_dim, 
            self.hyper_param.action_dim
        )
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        ###

        self.batch_step = 0

    def _batch_optim(self, transition: Transition):

        with torch.no_grad():
            a_t1 = self.target_actor(transition.next_state) + torch.randn(transition.action.size()) * self.hyper_param.sigma
            q_t1 = torch.min(self.target_critic(transition.next_state, a_t1), self.target_critic2(transition.next_state, a_t1))
            y_t = transition.reward + self.hyper_param.gamma * q_t1 * (1.0 - transition.done)

        ###

        q_t = self.critic(transition.state, transition.action)
        critic_loss: torch.Tensor = self.critic_loss_fn(q_t, y_t)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()   

        ###

        q2_t = self.critic2(transition.state, transition.action)
        critic2_loss: torch.Tensor = self.critic_loss_fn(q2_t, y_t)

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()   

        ###

        self.batch_step += 1
        if self.batch_step % self.hyper_param.actor_update_period == 0:
            a_t = self.actor(transition.state)
            actor_loss: torch.Tensor = -1 * torch.mean(self.critic(transition.state, a_t))

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self._sync_target_network(self.actor, self.target_actor)
            self._sync_target_network(self.critic, self.target_critic)
            self._sync_target_network(self.critic2, self.target_critic2)

        return critic_loss

    def update_episode(self, episode: int):
        self.batch_step = 0

    ### 