from elsa_tiago_fl.utils.utils import Singleton
import numpy as np
import torch
from tqdm import tqdm
from elsa_tiago_fl.utils.rl_utils import preprocess, get_custom_reward
import time
import random


def fl_evaluate(model, env, config):
    total_reward = []
    total_len_episode = []

    model.eval()

    for _ in tqdm(range(config.num_eval_episodes)):
        episode_reward, episode_length = 0.0, 0
        done = False
        observation= env.reset()
        while not done:
            state = preprocess(observation,model.multimodal,model.device)
            with torch.no_grad():
                action = model.select_action(
                        state,
                        config=config,
                        training=False,
                        action_space=env.action_space,
                    )

            act = model.get_executable_action(action)

            observation, reward, terminated, _= env.step(act) 
            custom_reward = float(get_custom_reward(env))
            reward += custom_reward

            episode_reward += reward
            done = terminated #or truncated
            episode_length += 1

        total_reward.append(episode_reward)
        total_len_episode.append(episode_length)

    avg_reward, std_reward = np.average(total_reward), np.std(total_reward)
    avg_len_episode, std_len_episode = np.average(total_len_episode), np.std(
        total_len_episode
    )

    return avg_reward, std_reward, avg_len_episode, std_len_episode


def client_evaluate(model, env, config):
    total_rewards = [[],[],[]]
    total_len_episode = []

    model.eval()

    for _ in tqdm(range(config.num_eval_episodes)):
        episode_reward, episode_reward_1, episode_reward_2, episode_length = 0.0, 0.0, 0.0, 0
        done = False
        observation= env.reset()
        while not done and episode_length<10:
            state = preprocess(observation,model.multimodal,model.device)
            with torch.no_grad():
                action = model.select_action(
                        state,
                        config=config,
                        training=False,
                        action_space=env.action_space,
                    )

            act = model.get_executable_action(action)

            observation, reward, terminated, _= env.step(act) 
            custom_reward, reward_1, reward_2 = get_custom_reward(env,True)
            reward+=custom_reward
            
            episode_reward += reward
            episode_reward_1 += reward_1
            episode_reward_2 += reward_2

            done = terminated #or truncated
            episode_length += 1

        total_rewards[0].append(episode_reward)
        total_rewards[1].append(episode_reward_1)
        total_rewards[2].append(episode_reward_2)


    avg_reward = np.average(total_rewards[0])
    avg_reward_1 = np.average(total_rewards[1])
    avg_reward_2 = np.average(total_rewards[2])

    return avg_reward, avg_reward_1, avg_reward_2


class Evaluator(metaclass=Singleton):
    def __init__(self):
        self.total_avg_rewards, self.total_std_rewards = [], []
        self.total_avg_episode_length, self.total_std_episode_length = [], []
        self.best_reward = 0
        self.best_epoch = 0

    def update_global_metrics(self, avg_reward, current_epoch):
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_epoch = current_epoch
            return True
        else:
            return False
