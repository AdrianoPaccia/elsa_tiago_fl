from models.model import BasicModel
from elsa_tiago_fl.utils.build_utils import build_model,build_optimizer
from elsa_tiago_fl.utils.utils_parallel import set_parameters_model
import pandas as pd
from elsa_tiago_fl.utils.evaluation_utils import fl_evaluate, client_evaluate
from elsa_tiago_gym.utils_ros_gym import start_env
from elsa_tiago_fl.utils.rl_utils import preprocess, get_custom_reward, cheat_action, transition_from_batch
import torch
from elsa_tiago_fl.utils.utils import parse_args, load_config, seed_everything, delete_files_in_folder
from elsa_tiago_gym.utils_parallel import launch_master_simulation,kill_simulations
import os
import time
import numpy as np
from tqdm import tqdm
import random
from elsa_tiago_fl.utils.rl_utils import BasicReplayBuffer,Transition,get_buffer_variance
from elsa_tiago_fl.utils.utils import tic,toc
from elsa_tiago_fl.utils.utils_parallel import save_weigths
import json


def main(config):

    # HYPERPARAMS ----------------------------------------------------------------------------------------------
    with_noise = True
    eps = 0.2
    n_envs = 50
    n_iter_per_env = 50
    env_codes = [x for x in range(n_envs)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print('Starting environments')
    env = start_env(env=config.env_name,
            speed = config.gz_speed,
            client_id = 0,
            max_episode_steps = config.max_episode_steps,
            multimodal = config.multimodal,
            random_init =True
    )
    time.sleep(5)
    env.env_kind = 'environments_4'


    ## START COLLECTION -------------------------------------------------------------------------------------------------
    print('Starting Collecting')
    #pick cube kind
    for env_code in env_codes:        
        trajectories = {}
        with tqdm(total=n_iter_per_env, desc=f'Environment {env_code}') as pbar:
            
            #rollout
            for i in range(n_iter_per_env):
                trajectory = []
                done = False
                step_i = 0
                env.init_environment(env_code)
                state = env.reset()
                state = preprocess(state, multimodal=False,device= device)
                while (not done) and (step_i < config.num_steps):
                    if with_noise:
                        if random.random < eps: #noisy action
                            action = [np.uniform(-1,1) for _ in range(4)]
                        else:
                            action = cheat_action(env)                            
                    next_state, reward, done, info = env.step(action)
                    custom_reward = float(get_custom_reward(env))
                    reward += custom_reward                                 #model.device
                    next_state = preprocess(next_state, multimodal=False,device= device)
                    transition = (
                        state.cpu().squeeze().tolist(),
                        action,
                        next_state.cpu().squeeze().tolist(),
                        [reward],
                        [done]
                    )
                    state = next_state
                    trajectory.append(transition)
                    step_i += 1
            pbar.update()
            key = str(i)+'_'+str(env_code)#+'_'+str(step_tot)
            trajectories[key] = trajectory

        file_path = f'datasets/traj_dataset_{env_code}_expert.json'
        save_dict(trajectories,file_path)           


def save_dict(my_dict,file_path):
    try:
        json_str = json.dumps(my_dict)
        with open(file_path, 'w') as file:
            file.write(json_str)
            file.close()
        print(f'Dict saved in {file_path}!')
        return True
    except:
        print('NOT saved!')
        return False

        

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    seed_everything(config.seed)
    config.gz_speed=0.001 #5
    launch_master_simulation(gui=config.gui)
   
    main(config)
    kill_simulations()



