from abc import ABC
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .transition import *

import os

class RLModel(ABC):

    ### 训练相关函数
    def take_action(self, state: np.ndarray) -> np.ndarray:
        '''
        执行动作 (训练模式下)
        '''
        ...

    def update_step(self, transition: Transition) -> None | float:
        '''
        按步更新模型
        '''
        ...

    def update_episode(self, episode: int):
        '''
        按片段更新模型
        '''
        ...

    ### 部署相关函数
    def take_action_deploy(self, state: np.ndarray) -> np.ndarray:
        '''
        部署模式下执行动作
        '''
        ...

class SampleModel(RLModel):
    def __init__(self, env: gym.Env) -> None:
       self.env = env
       pass

    def take_action(self, state: np.ndarray) -> np.ndarray:
        return self.env.action_space.sample()

    def update_step(self, transition: Transition) -> None | float:\
        return None

    def update_episode(self, episode: int):
        pass

    def take_action_deploy(self, state: np.ndarray) -> np.ndarray:
        return self.env.action_space.sample()

def train_rl(model: RLModel, env: gym.Env, name: str, comment: str, episode: int = 500, is_log: bool = True, vedio_record_gap: int = 100):
    writer = None
    if is_log:
        env = RecordEpisodeStatistics(env, buffer_length = 1)
        env = RecordVideo(
            env, 
            video_folder = "vedio", 
            name_prefix = name,
            episode_trigger = lambda x: (x + 1) % vedio_record_gap == 0
        )
        writer = SummaryWriter(comment = name + "_" + comment)

    for episode in tqdm(range(episode)):
        state, info = env.reset()
        done = False
        total_loss = 0

        while not done:
            
            # 完成一次状态转移
            action = model.take_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 更新模型
            transition = make_transition_from_numpy(state, action, next_state, reward, done)
            loss = model.update_step(transition)
            if loss != None:
                total_loss += loss

            state = next_state
        
        model.update_episode(episode)

        if writer != None:        
            writer.add_scalar(
                f"{name}/avg_loss",
                total_loss / info["episode"]["l"],
                episode
            )
            writer.add_scalar(
                f"{name}/return",
                info["episode"]["r"],
                episode
            )

    env.close()
    
    if writer != None:
        writer.close()

class RL_Teacher():
    def __init__(self, model: RLModel, name: str, comment: str, **env_kwargs) -> None:
        self.model = model
        self.env_kwargs = env_kwargs
        self.name = name
        self.comment = comment

    def train(self, episode: int = 500, is_log: bool = True, is_fix_seed: bool = False, last_episode_return: int | None = None, vedio_fold: str = "train_vedio", vedio_record_gap: int = 100) -> float:
        '''
        训练模型, 并尝试记录视频与学习曲线到 Tensorboard 上
        '''
        writer = None
        env = gym.make(**self.env_kwargs)
        env = RecordEpisodeStatistics(env, buffer_length = 1)
        total_return = 0

        if last_episode_return == None:
            last_episode_return = episode

        if is_log:
            env = RecordVideo(
                env, 
                video_folder = os.path.join(vedio_fold, self.name), 
                name_prefix = self.comment,
                episode_trigger = lambda x: (x + 1) % vedio_record_gap == 0
            )
            writer = SummaryWriter(comment = self.name + "_" + self.comment)

        for e in tqdm(range(episode)):
            if is_fix_seed:
                state, info = env.reset(seed = e)
            else:
                state, info = env.reset(seed = e)
            done = False
            total_loss = 0
            valid_loss_count = 0

            while not done:
                
                # 完成一次状态转移
                action = self.model.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # 更新模型
                transition = make_transition_from_numpy(state, action, next_state, reward, done)
                loss = self.model.update_step(transition)
                if loss != None:
                    total_loss += loss
                    valid_loss_count += 1

                state = next_state
            
            self.model.update_episode(e)

            if writer != None:        
                writer.add_scalar(
                    f"{self.name}/avg_loss",
                    total_loss / valid_loss_count,
                    e
                )
                writer.add_scalar(
                    f"{self.name}/return",
                    info["episode"]["r"],
                    e
                )
            if e > episode - last_episode_return:
                total_return += info["episode"]["r"]

        env.close()
        
        if writer != None:
            writer.close()
        
        return total_return / last_episode_return

    def test(self, test_times: int = 10, is_log_vedio: bool = True, vedio_fold: str = "test_vedio", vedio_record_gap: int = 2) -> float:
        '''
        测试模型, 将模型指定次数测试的平均总回报作为模型评分
        '''
        env = gym.make(**self.env_kwargs)
        env = RecordEpisodeStatistics(env, buffer_length = 1)
        if is_log_vedio :
            env = RecordVideo(
                env, 
                video_folder = os.path.join(vedio_fold, self.name), 
                name_prefix = self.comment,
                episode_trigger = lambda x: (x + 1) % vedio_record_gap == 0
            )
            
        total_reward: float = 0

        for episode in tqdm(range(test_times)):
            # 固定种子为 episode
            state, info = env.reset(seed = episode)
            done = False
            
            while not done:
                
                # 完成一次状态转移
                action = self.model.take_action_deploy(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state

            total_reward += info["episode"]["r"]

        env.close()
        return total_reward / test_times