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


num_to_str = {
    0:"green",
    1:"red",
    2:"yellow",
    3:"blue",
}

def main(config):

    env = start_env(env=config.env_name,
                speed = config.gz_speed,
                client_id = 0,
                max_episode_steps = config.max_episode_steps,
                multimodal = config.multimodal,
                random_init =False
    )

    env.env_kind = 'environments_0'
    num_init_pos = 10
    granularity = 0.05
    #get point in the table grid
    table_low=[0.40,-0.35,0.443669]
    table_high =[0.60,0.35,0.443669]
    gripper_low = [0.40,-0.3,0.65]
    gripper_high = [0.60, 0.3,0.85]

    n_x = int((table_high[0]-table_low[0]) / granularity) + 1
    n_y = int((table_high[1]-table_low[1]) / granularity) + 1
    x = np.linspace(table_low[0], table_high[0], n_x)  
    y = np.linspace(table_low[1], table_high[1], n_y)  
    X, Y = np.meshgrid(x, y)
    cube_pos = np.column_stack((X.flatten(), Y.flatten()))
    
    ## qua serve perche state è model.device ma non ha model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')

    env_codes = [3,2,1,0]

    print('\nSTARING with:')
    print(f'Gripper pos = {num_init_pos}\nCube kinds = {env_codes}\nCube pos = {len(cube_pos)}')

    trajectories = {}
    print('Starting Collecting')
    #pick cube kind
    for env_code in env_codes:
        #pick arm pos
        for i in range(num_init_pos):
            gripper_init_pos  = np.random.uniform(low=gripper_low, high=gripper_high, size=(3,)).tolist()
            gripper_init_pos.extend([0,np.pi/2,0])
            #pick cube pos
            with tqdm(total=len(cube_pos), desc=f'Iter {i} - Cube {num_to_str[env_code]}') as pbar:
                step_tot = 0
                for c_pos in cube_pos:
                    c_pose = [np.append(c_pos,[0.443669,0,0,0]).tolist()]
                    env.reset()
                    state = env.impose_configuration(gripper_init_pos, env_code, c_pose)
                    state = preprocess(state, multimodal=False,device= device)

                    #rollout
                    trajectory = []
                    done = False
                    step_i = 0
                    while (not done) and (step_i < config.num_steps):
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
                    step_tot +=1
                    key = str(i)+'_'+str(env_code)+'_'+str(step_tot)
                    trajectories[key] = trajectory

                file_path = 'traj_dataset_1.json'
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
    config.gz_speed=0.005
    launch_master_simulation(gui=config.gui)
   
    main(config)
    kill_simulations()



