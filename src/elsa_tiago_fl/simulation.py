from models.model import BasicModel
from elsa_tiago_fl.utils.build_utils import build_model
from elsa_tiago_fl.utils.utils_parallel import set_parameters_model, get_model_with_highest_score
import pandas as pd
from elsa_tiago_gym.utils_ros_gym import start_env
from elsa_tiago_fl.utils.rl_utils import preprocess, get_custom_reward
from elsa_tiago_fl.utils.utils import parse_args, load_config, seed_everything, delete_files_in_folder
from elsa_tiago_gym.utils_parallel import launch_master_simulation,kill_simulations
import os
import glob
import torch
import time
import rosgraph
import numpy as np
from tqdm import tqdm
import random


def is_roscore_running():
    try:
        rosgraph.Master('/rostopic').getPid()
        return True
    except rosgraph.MasterError:
        return False






def evaluate(model, env, config, num_episodes):
    total_reward = []
    total_len_episode = []

    model.eval()

    for _ in tqdm(range(num_episodes)):
        episode_reward, episode_length = 0.0, 1
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





def main(config):

    #build the model with the highest score
    model =  build_model(config)
    model = get_model_with_highest_score(model,config)
    #get the env
    env = start_env(env=config.env_name,
                speed = config.gz_speed,
                client_id = config.client_id,
                max_episode_steps = config.max_episode_steps,
                multimodal = config.multimodal,
                random_init = config.random_init
    )
    print(f'episode steps = {env.max_episode_steps}')
    avg_reward, std_reward, avg_episode_length, std_episode_length = evaluate(model, env, config, 10)
    
    print(f"Evaluation Reward: {avg_reward} +- {std_reward}")
    print(f"Evaluation steps: {avg_episode_length} +- {std_episode_length}")



if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    seed_everything(config.seed)
    #launch_master_simulation(gui=config.gui)
    #config.gz_speed =  None #0.005
    #config.client_id = input('Input the number of the client to test (inv for a random one): ')
    #if config.client_id=='':
    #    config.client_id =None
    config.client_id =None
    main(config)
    kill_simulations()



