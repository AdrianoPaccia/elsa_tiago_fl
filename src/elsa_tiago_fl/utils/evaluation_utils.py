from elsa_tiago_fl.utils.utils import Singleton
import numpy as np
import torch
from tqdm import tqdm
from elsa_tiago_fl.utils.rl_utils import preprocess
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



def evaluate(model, env, config, num_episodes):
    total_reward = []
    total_len_episode = []

    model.eval()

    for _ in tqdm(range(num_episodes)):
        episode_reward, episode_length = 0.0, 1
        done = False
        observation= env.reset()
        act = [0]*8
        while not done:
            state = preprocess(observation,model.multimodal,model.device)
            with torch.no_grad():
                action = model.select_action(
                        state,
                        config=config,
                        training=False,
                        action_space=env.action_space,
                    )

            #act = model.get_executable_action(action)
            act = [random.uniform(-1., 1.) for _ in range(7)]

            #act = [0.0*(-1 if episode_length%2==0 else 1)]*8
            if random.random() > 0.5:
                act.append(True)
            else:
                act.append(False)

            observation, reward, terminated, _= env.step(act) 

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
