import flwr as fl
from models.model import BasicModel
from utils.evaluation_utils import Evaluator
import torch
import multiprocessing as mp
from typing import Dict, Optional
from elsa_tiago_fl.utils.rl_utils import BasicReplayBuffer,Transition
import os
import gym
from collections import deque
from flwr.common import NDArrays, Scalar
from argparse import Namespace
from elsa_tiago_fl.utils.checkpoint import save_model
from elsa_tiago_fl.utils.utils_parallel import (
    set_parameters_model,
    get_parameters_from_model,
    weighted_average,
)
from gym.envs.registration import register
from elsa_tiago_fl.utils.logger import Logger
import copy
import time
import numpy as np
from elsa_tiago_fl.utils.evaluation_utils import fl_evaluate
from tqdm import tqdm
from elsa_tiago_fl.utils.communication_utils import log_communication


from elsa_tiago_fl.utils.mp_logging import set_logs_level
import pickle
from elsa_tiago_fl.utils.mp_utils import WorkerProcess,PolicyUpdateProcess,get_shared_params,ExperienceQueue,EvaluationProcess,ResultsList
import sys
#for setup ros env
import rospkg
import rospy
import subprocess
from elsa_tiago_gym.utils_parallel import launch_simulations,kill_simulations,set_velocity
from elsa_tiago_gym.utils_ros_gym import start_env


os.system('rm -r /tmp/ray/')
mp.set_start_method('spawn')

#setup logs managers
set_logs_level()

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
        self.replay_buffer = replay_buffer 
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

            # Load the replay buffer
            if self.replay_buffer is not None:
                print(f" - loading replay buffer ({len(client_data['replay_buffer'])} samples)")
                self.replay_buffer.memory = deque(
                    client_data['replay_buffer'], maxlen=self.replay_buffer.capacity
                )
            # Load the model data (steps_done, state_dict)
            print(f" - loading steps done ({client_data['model_steps_done']})")
            self.model.steps_done = client_data["model_steps_done"]

            # Load the optimizer state_dict
            print(f" - loading optimizer parameters")
            self.optimizer.load_state_dict(client_data["optimizer_state_dict"])
        




    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """
        This method calls the training routine of the model.
        At the end saves the memory related to the optimizer and the experience
        """


        # set the model paramaters 
        self.set_parameters(parameters)

        # Setup the model for training and retreinve the initial parameters
        init_parameters, init_frozen_parameters = self.model.setup_fl_training(self.optimizer)

        #setup logs managers
        set_logs_level()

        ## Multiprocessing:
        print(f'(FIT) Creating Multiple Processes (1 policy updater and {self.n_workers} workers)')
    
        # Initialize the manager and shared variables
        manager = mp.Manager()
        replay_queues = [ExperienceQueue(mp.Queue(maxsize=50),manager.RLock(),i) for i in range(self.n_workers)]  
        lock_SP = manager.RLock()
        termination_event = mp.Event()
        init_state_dict = copy.deepcopy(self.model.state_dict())
        initial_params_bytes = pickle.dumps(self.model.state_dict())
        shared_params = manager.Value('c', initial_params_bytes)


        # Start the policy updater process that simultaneusly trains the model
        updater_process = PolicyUpdateProcess(
                                           model = self.model,
                                           shared_params = shared_params,
                                           lock = lock_SP,
                                           logger=self.logger,
                                           optimizer = self.optimizer,
                                           replay_buffer = self.replay_buffer,
                                           replay_queues = replay_queues,
                                           local_file_name = self.client_local_filename,
                                           config = self.config,
                                           env = self.env,
                                           client_id = self.client_id,
                                           termination_event = termination_event,
                                           screen=False
                                           )
        updater_process.start()

        # Start all workers that collect experience
        workers = [WorkerProcess(worker_id=i,
                    model = copy.deepcopy(self.model), 
                    replay_queue = replay_queues[i],
                    shared_params = shared_params,
                    lock = lock_SP,
                    env = self.env,
                    client_id = self.client_id,
                    config = config,
                    termination_event = termination_event
                    ) for i in range(self.n_workers)]

        for worker in workers:
            worker.start()

        # Wait for all processes to finish
        for worker in workers:
            worker.join()
        updater_process.join()


        processes = [updater_process, *workers]
        print('Wait until all processes close')
        while any(p.is_alive() for p in processes):
            print("At least one process is still alive")
            time.sleep(1)        

        upd_parameters = get_shared_params(shared_params,lock_SP)


        # close all queues and other shared memory
        #for queue in replay_queues:
        #    queue.shutdown()
        manager.shutdown()
        


        ## After all the iterations:
        # Get model updates on the parameters
        agg_update = [
            w - w_0 for w, w_0 in zip(list(upd_parameters.values()), init_parameters)
        ] 

        # Send weights of NON-Frozen layers
        upd_weights = [
            torch.add(agg_upd, w_0).cpu()
            for agg_upd, w_0, is_frozen in zip(
                agg_update, copy.deepcopy(init_parameters), init_frozen_parameters
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


        return upd_weights, 1, {}

    

    def evaluate(self, parameters, config: Namespace):
        """
        This method calls the evaluating routine of the model.
        At the end, collects the merics and stores the model states. 
        """
        print('Starting the evaluation of the entire Fl_model')

        self.set_parameters(parameters)

        #setup logs managers
        set_logs_level()

        ## Multiprocessing:
        # Initialize the manager and shared variables
        manager = mp.Manager()
        results_list = manager.list([0, 0, 0]) 
        
        # Multiple evaulation process
        '''
        results_list = ResultsList(manager.list(),manager.RLock())

        workers = [EvaluationProcess(
            model = copy.deepcopy(self.model), 
            shared_results = results_list,
            env = self.env,
            client_id = self.client_id,
            worker_id = i,
            config = config,
        )  for i in range(1)]

        for worker in workers:
            worker.start()
        
        with tqdm(total=tot_iter, desc=f"Evaluation") as pbar:
            while pbar.n < pbar.total:
                n = results_list.size() - pbar.n
                pbar.update(n)
                time.sleep(10)

        for worker in workers:
            worker.join()'''

        evaluation_process = EvaluationProcess(
            model = copy.deepcopy(self.model), 
            shared_results = results_list,
            env = self.env,
            client_id = None,   #choses randomly each iteration which environment to choose
            config = config,
        )

        evaluation_process.start()
        evaluation_process.join()

        avg_reward, std_reward, avg_episode_length, std_episode_length = results_list.get_score()
        manager.shutdown()


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


    def set_parameters(self, parameters):
        set_parameters_model(self.model, parameters)

    def get_parameters(self, config):
        get_parameters_from_model(self.model)

