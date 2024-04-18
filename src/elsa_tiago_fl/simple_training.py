from models.model import BasicModel
from elsa_tiago_fl.utils.build_utils import build_model,build_optimizer
from elsa_tiago_fl.utils.utils_parallel import set_parameters_model
import pandas as pd
from elsa_tiago_fl.utils.evaluation_utils import fl_evaluate, client_evaluate
from elsa_tiago_gym.utils_ros_gym import start_env
from elsa_tiago_fl.utils.rl_utils import preprocess, get_custom_reward, cheat_action, transition_from_batch

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
from elsa_tiago_fl.utils.rl_utils import BasicReplayBuffer,Transition,get_buffer_variance
import wandb
from elsa_tiago_fl.utils.utils import tic,toc
from elsa_tiago_fl.utils.utils_parallel import save_weigths
import math


def save_weigths(model,config):
    save_dir = os.path.join(config.save_dir, 'weigths', 'client'+str(config.client_id))
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir,
        config.model_name + '_' + config.env_name  
        )
    if config.multimodal:
        save_name = save_name + '_multimodal'
    else:
        save_name = save_name + '_input' + str(config.input_dim)
    save_name = save_name + '_'+ str(int(avg_reward)) + '.pth'
    torch.save(model.state_dict(), save_name)
    print(f'Model Saved in: {save_name} ')


def train(model, env, replay_buffer, config):
    model.train()
    model.total_loss = 0
    config.min_len_replay_buffer = 1000



    ## PREFILLING ------------------------------------------------------------------
    timesteps = []
    with tqdm(total=config.min_len_replay_buffer, desc=f'prefilling ERP') as pbar:
        while pbar.n < pbar.total:
            ##prefilling
            tic()
            state = env.reset()

            state = preprocess(state, multimodal=False,device=model.device)
            done = False
            steps = 0

            #config.num_steps
            while (not done) and (steps < config.num_steps) and (pbar.n < pbar.total):
                action = model.select_action(state,training=True,env=env)
                act =model.get_executable_action(action)

                next_state, reward, done, info = env.step(act)
                custom_reward, reward_1, reward_2 = get_custom_reward(env, True)
                reward += custom_reward

                next_state = preprocess(next_state, multimodal=False,device=model.device)
                transition = Transition(
                    state.cpu(),
                    torch.tensor(action, dtype=torch.float32).unsqueeze(0),
                    next_state.cpu(),
                    torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(done, dtype=torch.float32).unsqueeze(0)
                )
                
                replay_buffer.push(transition)
                pbar.update()
                timesteps.append(toc())
                steps += 1
    avg = np.mean(timesteps)
    std = math.sqrt(np.var(timesteps))
    #add the line
    line=f'\nsimple_env - speed up ({config.gz_speed}) - max displacement = {0.05}: {config.min_len_replay_buffer} samples in {sum(timesteps)}s ==> {avg}+-{std}'
    with open("measurements.txt", "a") as file:
        file.write(line)
    input(line) 
    total_reward = []
    total_len_episode = []



    ## TRAININNG LOOP ------------------------------------------------------------------
    with tqdm(total=1000, desc=f'training episodes') as pbar:
        for episode in range(1000):
            model.episode_loss = [0,0,0]
            observation = env.reset()
            state = preprocess(observation,model.multimodal,model.device)
            i_step = 0
            done = False
            while not done and i_step < config.num_steps:
                #action = model.select_action(state,training=True,env=env)
                #act =model.get_executable_action(action)
                action = [np.uniform(-1,1) for _ in range(4)]
                next_state, reward, done, info = env.step(act)

                custom_reward, reward_1, reward_2 = get_custom_reward(env, True)
                reward += custom_reward
                next_state = preprocess(next_state, multimodal=False,device=model.device)
         
                transition = Transition(
                    state.cpu(),
                    torch.tensor(action, dtype=torch.float32).unsqueeze(0),
                    next_state.cpu(),
                    torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(done, dtype=torch.float32).unsqueeze(0)
                )
                print('transition: ',transition)

                transition.to_cpu()
                replay_buffer.push(transition) 

                # update the network
                if i_step%config.updating_freq == 0:
                    batch = replay_buffer.sample(config.batch_size)
                    for b in batch:
                        b.to(model.device)
                    batch = transition_from_batch(batch)
                    loss = model.training_step(batch)

                state = next_state
                i_step += 1
            pbar.update()

            #evaluate the fl_round
            #avg_reward,std_reward,avg_episode_length,std_episode_length = fl_evaluate(model, env, config)
            avg_reward, avg_reward_1, avg_reward_2 = client_evaluate(model, env, config)
            print("avg_stuff: ", avg_reward, avg_reward_1, avg_reward_2)

            input()

            #get the dispertion of datapoints
            points = np.array([t.state.cpu().squeeze()[-3:] for t in list(replay_buffer.memory)])
            dispertion = get_buffer_variance(points)  

            log_dict = {
                    "Episode policy loss": model.episode_loss[1],
                    "Episode value loss": model.episode_loss[2],
                    "Evaluation avg reward": avg_reward,
                    "epsilon": model.eps_linear_decay(),
                    "dispertion": dispertion,
                    "buffer capacity":replay_buffer.get_capacity(),
                }       
            wandb.log(log_dict)

            #save weigths
            #where = save_weigths(model,avg_reward,config)



def main(config):
    config.client_id = None

    #build the model and the optimizer
    model = build_model(config)
    optimizer=build_optimizer(model, config),
    replay_buffer = BasicReplayBuffer(capacity=config.replay_buffer_size)  

    #get the env
    env = start_env(env=config.env_name,
                speed = config.gz_speed,
                client_id = config.client_id,
                max_episode_steps = config.max_episode_steps,
                multimodal = config.multimodal,
                random_init = config.random_init
    )

    train(model, env, replay_buffer, config)







if __name__ == "__main__":

    #parsing
    args = parse_args()
    config = load_config(args)
    seed_everything(config.seed)
    config.gz_speed = 0.005
    launch_master_simulation(gui=config.gui)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    wandb.init(project="robot simple training", config=config)

    main(config)
    kill_simulations()



