from elsa_tiago_fl.utils.utils import Singleton
import numpy as np
import torch
from tqdm import tqdm
from elsa_tiago_fl.utils.rl_utils import preprocess


def fl_evaluate(model, env, config):
    total_reward = []
    total_len_episode = []

    model.eval()

    for _ in tqdm(range(config.num_eval_episodes)):
        episode_reward, episode_length = 0.0, 0
        done = False
        state= env.reset()
        while not done:
            state = preprocess(state,model.multimodal,model.img_size,model.device)
            with torch.no_grad():
                #action,_ = model.select_action(state, training=True)
                action = model.select_action(
                        state,
                        config=config,
                        training=False,
                        action_space=env.action_space,
                    )

            if config.discrete_actions:
                state, reward, terminated, _= env.step(model.executable_act[action]) 
            else:
                state, reward, terminated, _= env.step(action.item())    

            #state, reward, terminated, _ = env.step(action)
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
