import os
import yaml
import random
import datetime
import argparse
import numpy as np
import torch
import rospy
import rospkg
from time import time
import inspect


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Robotics FL Baseline")

    # Required
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required= False,
        help="Path to yml file with model configuration.",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        required= False,
        help="Path to yml file with env configuration.",
    )

    parser.add_argument(
        "-n_workers",
        "--n_workers",
        type=int,
        required= False,
        help="Number of workers running in parallel.",
    )

    parser.add_argument(
        "-gui",
        "--gui",
        type=bool,
        default=False,
        help="Get all the environment visible on the screen.",
    )
    # Overwrite config parameters
    parser.add_argument(
        "-steps", "--num-steps", type=int, help="Number of training steps"
    )
    parser.add_argument("-bs", "--batch-size", type=int, help="DataLoader batch size.")
    parser.add_argument("--seed", type=int, help="Seed to allow reproducibility.")
    parser.add_argument("--save-dir", type=str, help="Seed to allow reproducibility.")

    # Flower
    parser.add_argument(
        "--sample_clients", type=int, help="Number of sampled clients during FL."
    )
    parser.add_argument("--num_rounds", type=int, help="Number of FL rounds.")
    parser.add_argument(
        "--iterations_per_fl_round",
        type=int,
        help="Number of iterations per provider during each FL round.",
    )
    parser.add_argument(
        "--providers_per_fl_round",
        type=int,
        help="Number of groups (providers) sampled in each FL Round.",
    )

    try:
        parser.add_argument(
            "-__name",
            type=str,
            help="Who is the trainer.",
        )
        parser.add_argument(
            "-__log",
            type=str,
            help="path to log.",
        )
    except:
        parser.add_argument(
            "__name",
            type=str,
            help="Who is the trainer.",
        )
        parser.add_argument(
            "__log",
            type=str,
            help="path to log.",
        )


    return parser.parse_args()


def check_config(config):
    (
        config.fl_parameters.sample_clients <= config.fl_parameters.total_clients,
        "Number of sampled clients ({:d}) can't be greater than total number of clients ({:d})".format(
            config.fl_parameters.sample_clients, config.fl_parameters.total_clients
        ),
    )

    if "save_dir" in config:
        if not config.save_dir.endswith("/"):
            config.save_dir = config.save_dir + "/"

        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
            os.makedirs(os.path.join(config.save_dir, "results"))
            os.makedirs(os.path.join(config.save_dir, "communication_logs"))

    experiment_date = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.experiment_name = "{:s}_{:s}__{:}".format(
        config.model_name, config.env_name, experiment_date
    )

    return True

    
def load_config(args):
    
    #model = rospy.get_param('/tiago_trainer/model')
    #env = rospy.get_param('/tiago_trainer/env')


    model,env,n_workers = args.model, args.env,args.n_workers
    # Get the path to the parameters.yaml file
    package_name = "elsa_tiago_fl"  # Replace with your actual package name
    pkg_path = rospkg.RosPack().get_path(package_name)
    envs_folder_path = os.path.join(pkg_path,'config/envs')
    models_folder_path = os.path.join(pkg_path,'config/models')
    env_config_path = os.path.join(envs_folder_path,env+'.yaml')
    model_config_path = os.path.join(models_folder_path,model+'.yaml')

    model_config = parse_config(yaml.safe_load(open(model_config_path, "r")), args)
    env_config = parse_config(yaml.safe_load(open(env_config_path, "r")), args)
    training_config = model_config.pop("training_parameters")

    # Keep FL and DP parameters to move it to a lower level config (config.fl_config / config.dp_config)
    fl_config = (
        model_config.pop("fl_parameters") if "fl_parameters" in model_config else None
    )
    fl_dp_keys = []
    # Update (overwrite) the config yaml with input args.
    if fl_config is not None:
        fl_config.update(
            {k: v for k, v in args._get_kwargs() if k in fl_config and v is not None}
        )
        fl_dp_keys.extend(fl_config.keys())

    # Merge config values and input arguments.
    config = {**env_config, **model_config, **training_config}
    config = config | {k: v for k, v in args._get_kwargs() if v is not None}

    # Remove duplicate keys
    #config.pop("model")
    #config.pop("env")
    [config.pop(k) for k in list(config.keys()) if (k in fl_dp_keys)]

    config = argparse.Namespace(**config)

    if fl_config is not None:
        config.fl_parameters = argparse.Namespace(**fl_config)

    # Set default seed
    if "seed" not in config:
        print("Seed not specified. Setting default seed to '{:d}'".format(42))
        config.seed = 42

    check_config(config)

    return config


def parse_config(config, args):
    # Import included configs.
    for included_config_path in config.get("includes", []):
        config = load_config(included_config_path, args) | config

    return config


def delete_files_in_folder(folder_path):
    try:
        # Get a list of all files in the specified folder
        file_list = os.listdir(folder_path)

        # Iterate through the files and delete each one
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_name}")

        print("All files deleted successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def tic():
    tic.t = time()

def toc(message=None):
    time_elapsed = time() - tic.t
    if message is None:
        message = inspect.currentframe().f_back.f_lineno
    #print(message, time_elapsed)
    tic.t = time()
    return time_elapsed