import flwr as fl
from models.model import BasicModel
from utils.evaluation_utils import Evaluator
import torch
import multiprocessing as mp
from typing import Dict, Optional
from elsa_tiago.utils.rl_utils import BasicReplayBuffer,Transition
import os
import gym
from collections import deque
from flwr.common import NDArrays, Scalar
from argparse import Namespace
from elsa_tiago.utils.checkpoint import save_model
from elsa_tiago.utils.utils_parallel import (
    set_parameters_model,
    get_parameters_from_model,
    weighted_average,
)
from gym.envs.registration import register
from elsa_tiago.utils.logger import Logger
import copy
import time
import numpy as np
from elsa_tiago.utils.evaluation_utils import fl_evaluate
from tqdm import tqdm
from elsa_tiago.utils.communication_utils import log_communication
import pickle
from elsa_tiago.utils.mp_utils import WorkerProcess,PolicyUpdateProcess,get_shared_params,ExperienceQueue
import sys
import logging

#for setup ros env
import rospkg
import rospy
from elsa_tiago_gym.utils import setup_env
import subprocess
from elsa_tiago_gym.utils_parallel import launch_simulations,kill_simulations,set_velocity


os.system('rm -r /tmp/ray/')
mp.set_start_method('spawn')


'''
For cleaning the /tmp from previous runs files:  rm -r /tmp/ray/
'''

class FlowerClientMultiprocessing(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: str,
        model: BasicModel,
        env: str,#Env,
        n_workers:int,
        evaluator: Evaluator,
        optimizer: torch.optim.Optimizer,
        logger,
        config,
        replay_buffer: Optional[BasicReplayBuffer] = None,
    ) -> None:

        self.n_workers = n_workers
        self.env = env

        self.model = model
        self.replay_buffer = BasicReplayBuffer(1000) #replay_buffer

        self.optimizer = optimizer
        self.evaluator = evaluator
        self.logger = logger
        self.logger.log_model_parameters(self.model)
        self.config = config
        self.client_id = client_id
        self.client_local_filename = os.path.join(
            "temp", f"memory_client_{int(self.client_id)}.pth"
        )
        if not os.path.exists(os.path.join("temp")):
            os.makedirs(os.path.join("temp"))

        if os.path.exists(self.client_local_filename):
            print(
                f"Loading data for client #{int(client_id)} from file: {self.client_local_filename}"
            )
            client_data = torch.load(self.client_local_filename)
            if self.replay_buffer is not None:
                self.model.steps_done = client_data["model_steps_done"]
                self.replay_buffer.memory = deque(
                    client_data["replay_buffer"], maxlen=self.replay_buffer.capacity
                )
            #self.optimizer.load_state_dict(client_data["optimizer_state_dict"])
        




    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """
        This method calls the training routine of the model.
        At the end saves the memory related to the optimizer and the experience
        """

        # Setup the model for training
        parameters, frozen_parameters = self.model.setup_fl_training(self.optimizer)

        # Initialize the manager and shared variables
        manager = mp.Manager()
        replay_queues = [ExperienceQueue(mp.Queue(maxsize=50),manager.RLock(),i) for i in range(self.n_workers)]  
        lock_SP = manager.RLock()
        termination_event = mp.Event()

        init_state_dict = copy.deepcopy(self.model.state_dict())

        initial_params_bytes = pickle.dumps(self.model.state_dict())
        shared_params = manager.Value('c', initial_params_bytes)

        print('Creating Multiple Prcesses')

        env_config = {
            'env_name':self.env,
            'multimodal':self.config.multimodal,
            'discrete':self.config.discrete_actions,
            'max_episode_steps':100
        }

        #set the sim velocity

        set_velocity(
            n = config.n_workers,
            speed = config.velocity
        )


        # Start the policy updater process that simultaneusly trains the model
        updater_process = PolicyUpdateProcess(
                                           model = self.model,
                                           shared_params = shared_params,
                                           lock = lock_SP,
                                           logger=self.logger,
                                           optimizer = self.optimizer,
                                           replay_buffer = self.replay_buffer,
                                           replay_queues = replay_queues,
                                           #batch_queue = batch_queue,
                                           config = self.config,
                                           env = self.env,#gym.make(id=self.env,env_code=self.client_id,max_episode_steps=100),
                                           client_id = self.client_id,
                                           termination_event = termination_event
                                           )
        updater_process.start()

        # Start all workers that collect experience
        workers = [WorkerProcess(worker_id=i,
                    model = copy.deepcopy(self.model), 
                    replay_queue = replay_queues[i],
                    shared_params = shared_params,
                    lock = lock_SP,
                    env = self.env,#gym.make(id=self.env,env_code=self.client_id,max_episode_steps=100),
                    env_config = env_config,
                    client_id = self.client_id,
                    config = config,
                    termination_event = termination_event
                    ) for i in range(self.n_workers)]

        for worker in workers:
            worker.start()

        set_velocity(self.config.n_workers,0.007)
        print(f'gz physics spedd at {0.007}')

        # Wait for all processes to finish
        updater_process.join()
        for worker in workers:
            worker.join()

        upd_parameters = get_shared_params(shared_params,lock_SP)

        
        ## After all the iterations:
        # Get model update
        agg_update = [
            w - w_0 for w, w_0 in zip(list(upd_parameters.values()), parameters)
        ] 
        # Send weights of NON-Frozen layers.
        upd_weights = [
            torch.add(agg_upd, w_0).cpu()
            for agg_upd, w_0, is_frozen in zip(
                agg_update, copy.deepcopy(parameters), frozen_parameters
            )
            if not is_frozen
        ]  
        # Store only communicated weights (sent parameters).
        log_communication(
            federated_round=config.current_round,
            sender=self.client_id,
            receiver=-1,
            data=upd_weights,
            log_location=self.logger.comms_log_file,
        )  
        
        print(
            f"Saving Memory of Client #{int(self.client_id)} to file: {self.client_local_filename}"
        )
        if self.replay_buffer is not None:
            torch.save(
                {
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "model_steps_done": self.model.steps_done,
                    "replay_buffer": [n_tuple for n_tuple in self.replay_buffer.memory],
                },
                self.client_local_filename,
            )
        else:
            torch.save(
                {
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                self.client_local_filename,
            )
        return upd_weights, 1, {}
    
    def evaluate(self, parameters, config: Namespace):
        """
        This method calls the evaluating routine of the model.
        At the end, collects the merics and stores the model states. 
        """
        set_parameters_model(self.model, parameters)
        setup_env(self.env, config.velocity)
        env = gym.make(self.env)

        avg_reward, std_reward, avg_episode_length, std_episode_length = fl_evaluate(
            self.model, env, config
        )
        is_updated = self.evaluator.update_global_metrics(
            avg_reward, config.current_round
        )
        self.logger.log_val_metrics(
            avg_reward,
            std_reward,
            avg_episode_length,
            std_episode_length,
            update_best=is_updated,
        )
        save_model(
            self.model, config.current_round, update_best=is_updated, kwargs=config
        )
        print(f"Evaluation Reward: {avg_reward} +- {std_reward}")

        return (
            float(0),
            config.num_eval_episodes,
            {
                "avg_reward": float(avg_reward),
                "std_reward": float(std_reward),
                "avg_episode_length": float(avg_episode_length),
                "std_episode_length": float(std_episode_length),
            },
        )
    
