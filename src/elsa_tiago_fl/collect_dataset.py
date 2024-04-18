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
from collections import namedtuple
from elsa_tiago_gym.utils import generate_random_config_2d

TrajectoryDescription = namedtuple('TrajectoryDescription', ['cube_init_poses', 'gripper_init_pose', 'type_target', 'outcome'])

def main(config):

    # HYPERPARAMS ----------------------------------------------------------------------------------------------
    random_init = False
    with_noise = False
    eps = 0.2
    n_envs = 3
    n_iter_per_env = 10 #1000
    env_codes = [0,1,2]
    #env_codes.extend([x for x in range(17,n_envs)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Starting with environments: ',env_codes)
    env = start_env(env=config.env_name,
            speed = config.gz_speed,
            client_id = 0,
            max_episode_steps = config.max_episode_steps,
            multimodal = config.multimodal,
            random_init =random_init
    )
    time.sleep(5)
    env.env_kind = 'environments'

    ## START COLLECTION -------------------------------------------------------------------------------------------------
    print('Starting Collecting')
    #pick cube kind
    for env_code in env_codes:        
        trajectories = {}
        descriptions = {}
        env_name = give_name_traj(len(env.model_state.cubes), env.target_color)
        env.env_code = env_code

        ## ROLLOUT
        with tqdm(total=n_iter_per_env, desc=f'Environment {env_code}') as pbar:
            for i in range(n_iter_per_env):
                trajectory = []
                done = False
                step_i = 0

                #reset the environment and impose a new trajectory
                env.reset()
                gipper_pose, cube_poses = get_new_conf(env,5)
                state = env.impose_configuration(
                    gipper_pose = gipper_pose,
                    cube_poses = cube_poses,
                    env_code = env_code
                )
                
                #state = preprocess(state, multimodal=False,device= device)
                while (not done) and (step_i < config.num_steps):
                    if with_noise:
                        if random.random() < eps: 
                            action = [np.uniform(-1,1) for _ in range(4)]
                        else:
                            action = cheat_action(env) 
                    else:
                        action = cheat_action(env)                       
                    next_state, reward, done, info = env.step(action)
                    custom_reward = float(get_custom_reward(env))
                    reward += custom_reward                                 #model.device
                    #next_state = preprocess(next_state, multimodal=False,device= device)
                    transition = (
                        state,#.cpu().squeeze().tolist(),
                        action,
                        next_state,#.cpu().squeeze().tolist(),
                        [reward],
                        [done]
                    )
                    state = next_state
                    trajectory.append(transition)
                    step_i += 1
                pbar.update()
                key = str(i) 
                trajectories[key] = trajectory
                desc_i = TrajectoryDescription(
                    cube_init_poses = cube_poses,
                    gripper_init_pose = gipper_pose,
                    type_target = env.type_target,
                    outcome = '1' if reward>0 else '0'
                )
                descriptions[key] = desc_i
            #if i%10 == 0:
            file_path_traj = f'datasets/trajectories/{env_name}_{n_iter_per_env}samples.json'
            save_dict(trajectories,file_path_traj)           
            file_path_desc = f'datasets/descriptions/{env_name}_{n_iter_per_env}samples.json'
            save_dict(descriptions,file_path_desc)           


def save_dict(data:dict,file_path:str):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
        print(f'Saved in {file_path}!')
    except:
        print('NOT saved!')
        return Exception


def give_name_traj(n_cubes, color):
    return str(n_cubes) + '_' + color

def get_new_conf(env,n_cubes=5):
    c_pos = generate_random_config_2d(
                low = [0.45,-0.25],
                high = [0.55,0.25],
                n_cubes = n_cubes,
                dist = 0.1,
                threshold=0.01
                )
    c_poses = []
    for c in c_pos:
        c.extend([0.44,0,0,0])
        c_poses.append(c)
    x, y, z  = [np.random.uniform(low, high) for low, high in zip([0.40,-0.2,0.8], [0.5,0.2,0.85])]
    g_pose = [x, y, z, 0, np.radians(90), 0]
    return g_pose, c_poses
        

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    seed_everything(config.seed)
    config.gz_speed=0.001 #5
    launch_master_simulation(gui=config.gui)
   
    main(config)
    kill_simulations()



