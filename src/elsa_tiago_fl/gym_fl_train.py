#! /usr/bin/env python
from argparse import Namespace
from collections import deque
from typing import Dict, Optional
import os
import gym
import torch
import flwr as fl
import numpy as np
from flwr.common import NDArrays, Scalar
from gym import Env
from models.model import BasicModel
from elsa_tiago_fl.utils.build_utils import build_model, build_optimizer
from elsa_tiago_fl.utils.checkpoint import save_model
from elsa_tiago_fl.utils.evaluation_utils import fl_evaluate, Evaluator
from elsa_tiago_fl.utils.logger import Logger
from elsa_tiago_fl.utils.rl_utils import BasicReplayBuffer,TrajectoryHolder
from elsa_tiago_fl.utils.utils import parse_args, load_config, seed_everything, delete_files_in_folder
from elsa_tiago_fl.utils.utils_parallel import (
    set_parameters_model,
    get_parameters_from_model,
    weighted_average,
    get_model_with_highest_score,
)
import sys
import rospkg
from multiprocess_fl import FlowerClientMultiprocessing
import time
import subprocess
from elsa_tiago_gym.utils_parallel import launch_simulations,kill_simulations,set_velocity
import wandb




def main() -> None:
    args = parse_args()
    config = load_config(args)
    seed_everything(config.seed)

    #launch the simulation environments
    launch_simulations(config.n_workers, gui=config.gui)

    # Delete previous memory files
    pkg_path = rospkg.RosPack().get_path("elsa_tiago_fl")
    delete_files_in_folder(os.path.join(pkg_path,"src/elsa_tiago_fl/temp"))

    # Set `MASTER_ADDR` and `MASTER_PORT` environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9957"

    # Create model
    model = get_model_with_highest_score( build_model(config),config,True)
    params = get_parameters_from_model(model)

    def get_config_fn():
        """Return a function which returns custom configuration."""

        def custom_config(server_round: int):
            """Return evaluate configuration dict for each round."""
            config.current_round = server_round
            return config

        return custom_config
    

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=config.fl_parameters.sample_clients
        / config.fl_parameters.total_clients,
        fraction_evaluate=0.1,  # Sample only 1 client for evaluation
        min_fit_clients=config.fl_parameters.sample_clients,  # Never sample less than N clients for training
        min_evaluate_clients=1,  # Sample only 1 client for evaluation.
        min_available_clients=config.fl_parameters.sample_clients,  # Wait until N clients are available
        fit_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        on_fit_config_fn=get_config_fn(),  # Log path hardcoded according to /save dir
        on_evaluate_config_fn=get_config_fn(),
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if config.device == "cuda":
        client_resources = {"num_gpus": 1, "num_cpus": 1}  # TODO Check number of GPUs

    def client_fn(client_id):
    
        """Create a Flower client representing a single organization."""
        logger = Logger(config=config)


        """Check if we need a replay buffer"""
        if hasattr(config, "replay_buffer_size"):
            r_buffer = BasicReplayBuffer(capacity=config.replay_buffer_size)
        else:
            r_buffer = None

        """Get the environment"""
        fl_client = FlowerClientMultiprocessing(
            client_id=client_id,
            model=model,
            env=config.env_name,
            n_workers = config.n_workers,
            replay_buffer=r_buffer,
            evaluator=Evaluator(),
            optimizer=build_optimizer(model, config=config),
            logger=logger,
            config=config,
        )

        return fl_client

    # Start simulation
    torch.cuda.empty_cache()
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.fl_parameters.total_clients,
        config=fl.server.ServerConfig(num_rounds=config.fl_parameters.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        # ray_init_args={"local_mode": True}  # run in one process to avoid zombie ray processes
    )


if __name__ == "__main__":
    # login to wandb
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    main()

    #kill the simulations on exit
    kill_simulations()


    


