from models.model import BasicModel
from elsa_tiago_fl.utils.build_utils import build_model
from elsa_tiago_fl.utils.utils_parallel import set_parameters_model
import pandas as pd
from elsa_tiago_gym.utils_ros_gym import start_env
from elsa_tiago_fl.utils.rl_utils import preprocess, get_custom_reward
from elsa_tiago_fl.utils.utils import parse_args, load_config, seed_everything, delete_files_in_folder
from elsa_tiago_gym.utils_parallel import launch_master_simulation,kill_simulations
import os
import glob
import torch
from elsa_tiago_fl.utils.evaluation_utils import fl_evaluate, evaluate
import time
import rosgraph

def is_roscore_running():
    try:
        rosgraph.Master('/rostopic').getPid()
        return True
    except rosgraph.MasterError:
        return False


def get_model_with_highest_score(config):
    model = build_model(config)

    # Construct the pattern to match saved model files
    pattern = os.path.join(config.save_dir, 'weigths','client'+str(config.client_id),
                           f"{config.model_name}_{config.env_name}")
    if config.multimodal:
        pattern += '_multimodal'
    else:
        pattern += f'_input{config.input_dim}'
    
    pattern += '_*.pth'

    # List all matching model files
    model_files = glob.glob(pattern)

    try:
        # Extract scores from file names
        scores = [float(os.path.basename(file).split('_')[-1].split('.pth')[0]) for file in model_files]

        # Find the index of the file with the highest score
        highest_score_idx = scores.index(max(scores))

        # Load model parameters from the file with the highest score
        highest_score_model_file = model_files[highest_score_idx]
        model.load_state_dict(torch.load(highest_score_model_file))
        print(f'Loaded the model parameters: {highest_score_model_file}')
        return model

    except Exception as e:
        raise RuntimeError("No model parameters available!")

def main(config):

    #build the model with the highest score
    model = get_model_with_highest_score(config)


    #while not is_roscore_running():
    #    print('waiting for the core')
    #    time.sleep(1)

    #get the env
    env = start_env(env=config.env_name,
                speed = config.velocity,
                client_id = config.client_id,
                max_episode_steps = 100, #config.max_episode_steps,
                multimodal = config.multimodal
    )
    avg_reward, std_reward, avg_episode_length, std_episode_length = evaluate(model, env, config, 10)
    
    print(f"Evaluation Reward: {avg_reward} +- {std_reward}")
    print(f"Evaluation steps: {avg_episode_length} +- {std_episode_length}")



if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    seed_everything(config.seed)
    config.velocity = 0.0025

    launch_master_simulation(speed=config.velocity, gui=config.gui)

    config.client_id = input('Input the number of the client to test (inv for a random one): ')
    if config.client_id=='':
        config.client_id =None
    main(config)
    kill_simulations()



