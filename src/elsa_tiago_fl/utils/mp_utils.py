import multiprocessing as mp
import time
import gym
import pickle
import copy
from models.model import BasicModel
from elsa_tiago_fl.utils.rl_utils import Transition,preprocess, get_custom_reward
import logging
import torch
from tqdm import tqdm
from elsa_tiago_fl.utils.evaluation_utils import fl_evaluate, Evaluator
from elsa_tiago_fl.utils.utils import tic,toc
import os
import rospy
import rospkg
from elsa_tiago_gym.utils_ros_gym import start_env
from elsa_tiago_gym.utils_parallel import set_sim_velocity,kill_simulations
from elsa_tiago_fl.utils.mp_logging import set_logs_level
import numpy as np
from elsa_tiago_fl.utils.rl_utils import transition_from_batch





DEBUGGING = True
#setup logs managers
#set_logs_level()
if DEBUGGING:
    logging.basicConfig(level=logging.DEBUG)
    logger_debug = mp.get_logger()



class PolicyUpdateProcess(mp.Process):
    """
    This is the process that every 10sec updates the model wieights getting
        the experience from the replay buffer 
    """
    def __init__(self, model, lock, logger, optimizer, shared_params, replay_buffer, local_file_name, replay_queues, config, env, client_id, termination_event, screen=False):
        super(PolicyUpdateProcess, self).__init__()
        mp.log_to_stderr(logging.ERROR)

        self.model = model
        self.shared_params = shared_params
        self.lock = lock
        self.logger = logger
        self.replay_buffer = replay_buffer
        self.replay_queues = replay_queues 
        self.config = config
        self.env = env
        self.termination_event =termination_event
        self.cnt=0
        self.screen = screen
        self.optimizer = optimizer
        self.client_id = client_id
        self.local_file_name = local_file_name
        

    def run(self):
        log_debug('ready to update polices!',self.screen)
        # setup the eval environment
        env = start_env(env=self.env,
                speed = self.config.gz_speed,
                client_id = self.client_id,
                max_episode_steps = self.config.max_episode_steps,
                multimodal = self.config.multimodal
        )

        self.model.train()

        # wait to for the workers to start publishing
        ready = False
        while not ready:
            for queue in self.replay_queues:
                if (queue.size()>0):
                    ready = True

        # prefilling of the buffer at first
        tic()
        with tqdm(total=self.config.min_len_replay_buffer, 
            initial = np.clip(len(self.replay_buffer.memory),0,self.config.min_len_replay_buffer),
            desc="Filling Replay Buffer") as pbar:
            while pbar.n < pbar.total:
                n = self.get_transitions()
                pbar.update(n)
        t = round(toc(),3)
        log_debug(f'filling the {len(self.replay_buffer.memory)} exp required {t}sec, giving {len(self.replay_buffer.memory)/t} transition/sec',self.screen)
        
        for train_iter in range(self.config.fl_parameters.iterations_per_fl_round):
            with tqdm(total=self.config.train_steps, desc=f"Training iteration {train_iter} client #{self.client_id}") as pbar:
                while pbar.n < pbar.total:
                    # get available trasitions
                    self.get_transitions()
                    batch = self.get_batch()
                    #convert a batch of transitions in a transition-batch
                    batch = transition_from_batch(batch)

                    # make 1 training iteration and update the shared weigths
                    loss = self.model.training_step(batch)
                    self.send_shared_params(self.model.state_dict())

                    pbar.update()

            # log the episode loss
            self.model.log_recap('episode',self.logger)
            log_debug(f"END fl_round - Capacity at {round(self.replay_buffer.get_capacity()*100,0)}%",self.screen)

            #evaluate the fl_round
            (avg_reward,std_reward,avg_episode_length,std_episode_length,) = fl_evaluate(self.model, env, self.config)
            log_dict = {
                "Avg Reward (during training) ": avg_reward,
                "Std Reward (during training) ": std_reward,
            }
            self.logger.logger.log(log_dict)

            #save weigths
            save_dir = os.path.join(self.config.save_dir, 'weigths', 'client'+str(self.client_id))
            os.makedirs(save_dir, exist_ok=True)

            save_name = os.path.join(save_dir,
                self.config.model_name + '_' + self.config.env_name  
                )
            if self.config.multimodal:
                save_name = save_name + '_multimodal'
            else:
                save_name = save_name + '_input' + str(self.config.input_dim)
            save_name = save_name + '_'+ str(int(avg_reward)) + '.pth'
            torch.save(self.model.state_dict(), save_name)
            log_debug(f'Model Saved in: {save_name} ',self.screen)

            
        #log the avg round loss
        self.model.log_recap('round',self.logger)
        #self.termination_event.set()

        self.end_process()


    def send_shared_params(self,state_dict):
        try:
            state_dict_copy = copy.deepcopy(state_dict)
            new_params_bytes = pickle.dumps(state_dict_copy)
            with self.lock:
                self.shared_params.value = new_params_bytes
        except Exception as e:
            logging.error(f"Error sending shared params: {e}")
            return None

        
    def get_batch(self):
        try:
            #batch = self.batch_queue.get()
            batch = copy.deepcopy(self.replay_buffer.sample(self.config.batch_size))
            for b in batch:
                b.to(self.model.device)
            return batch
        except Exception as e:
            logging.error(f"Error getting batches: {e}")
            return None

    def get_transitions(self):
        try:
            n_trans = 0
            for queue in self.replay_queues:
                while not queue.empty():
                    transition = queue.get()
                    self.replay_buffer.push(transition) 
                    n_trans +=1
            return n_trans

        except Exception as e:
            logging.error(f"Error getting transitions: {e}")
            return None

    def end_process(self):
        """
        save the replays and send the termination event
        """
        self.termination_event.set()
        if self.replay_buffer is not None:
            torch.save(
                {
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "model_steps_done": self.model.steps_done,
                    "replay_buffer":  self.replay_buffer.memory,#[n_tuple for n_tuple in self.replay_buffer.memory],
                },
                self.local_file_name,
            )
        else:
            torch.save(
                {
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                self.local_file_name,
            )



class ExperienceQueue:
    def __init__(self,queue,lock,n):
        self.queue = queue
        self.lock = lock
        self.n = n
    def empty(self):
        return self.queue.empty()
    def rollout(self):
        datas = []
        with self.lock:
            while not self.queue.empty():
                data = self.queue.get()
                datas.append(data) 
        return datas
    def put(self,data):
        if not self.queue.full():
            with self.lock:
                self.queue.put(data)
    def get(self):
        with self.lock:
            return self.queue.get()
    def size(self):
        return self.queue.qsize()
    #def shutdown(self):
    #    while not self.queue.empty():
    #        self.queue.get()
    #    self.queue.close()


        
class WorkerProcess(mp.Process):
    """
    This process collects the experience from the environment using the latest model parameters
    """
    def __init__(self, worker_id, model: BasicModel, replay_queue, shared_params, lock, env, client_id,config, termination_event,screen=False):
        super(WorkerProcess, self).__init__()
        self.worker_id = worker_id
        self.model = model
        self.replay_queue = replay_queue
        self.shared_params = shared_params
        self.env = env
        self.client_id = client_id
        self.config = config
        self.termination_event = termination_event
        self.lock = lock
        self.screen =screen


    def run(self):
        log_debug('ready to get experiences!',self.screen)
        # get connected to one of the ros ports 
        self.env = start_env(env=self.env,
                speed = self.config.gz_speed,
                client_id = self.client_id,
                max_episode_steps = self.config.max_episode_steps,
                multimodal = self.config.multimodal,
                ros_uri = "http://localhost:1135" + str(self.worker_id) + '/',
                gz_uri = "http://localhost:1134" + str(self.worker_id) 
        )
        tot_step =0


        while not self.termination_event.is_set() and not rospy.is_shutdown():

            # get the new params of the model
            state_dict = get_shared_params(self.shared_params,self.lock)
            if state_dict is not None:
                self.model.load_state_dict(state_dict)
            else:
                log_debug('could NOT load new parameters',self.screen)
            
            self.model.train()

            # Explore the environment using the current policy
            observation = self.env.reset()
            state = preprocess(observation,self.model.multimodal,self.model.device)
            i_step = 0
            done = False
            while not done and i_step < self.config.num_steps:
                action = self.model.select_action(
                        state,
                        config=self.config,
                        training=True,
                        action_space=self.env.action_space,
                    )
                
                act = self.model.get_executable_action(action)

                #if self.config.discrete_actions:
                #    act = self.model.executable_act[action]
                #else:
                #    act = action.numpy()

                observation, reward, terminated, _= self.env.step(act) 
                custom_reward = float(get_custom_reward(self.env, -0.5, -0.5))
                #log_debug(f'custom_reward = {custom_reward}',True)
                reward += custom_reward

                done = terminated #or truncated

                next_state = preprocess(observation,self.model.multimodal,self.model.device)
                action_tsr = action.to(torch.float32).unsqueeze(0)
                reward_tsr = torch.tensor([reward], device=self.model.device)
                done_tsr = torch.tensor([done], dtype=float,device=self.model.device)

                # Put the transition in the queue 
                self.send_transition(state,action_tsr,next_state,reward_tsr,done_tsr)
 
                state = next_state
                i_step += 1
                tot_step +=1
                if self.termination_event.is_set():
                    break




    def send_transition(self,state,action,next_state,reward,done):
        try:
            state_clone = state.clone()
            action_clone = action.clone()
            next_state_clone = next_state.clone()
            reward_clone = reward.clone()
            done_clone = done.clone()
            transition = Transition(state_clone, action_clone, next_state_clone,reward_clone,done_clone)
            transition.to_cpu()
            self.replay_queue.put(transition)

            log_debug('sent a transition - queue size = {:}!'.format(self.replay_queue.size()),self.screen)

        except Exception as e:
            logging.error(f"Error getting shared params: {e}")
            return None


class ResultsList():
    def __init__(self,array,lock):
        self.array = array
        self.lock = lock

    def store(self,score,length):
        with self.lock:
            self.array.append((score,length))

    def get_score(self):
        total_reward = []
        total_len_episode = []
        with self.lock:
            results=self.array
        for item in results:
            reward, len_episode = item
        total_reward.append(reward)
        total_len_episode.append(len_episode)

        avg_reward, std_reward = np.average(total_reward), np.std(total_reward)
        avg_len_episode, std_len_episode = np.average(total_len_episode), np.std(
            total_len_episode
        )
        return avg_reward, std_reward, avg_len_episode, std_len_episode

    def size(self):
        return len(self.array)


class EvaluationProcess(mp.Process):
    """
    This process collects the experience from the environment using the latest model parameters
    """
    def __init__(self, model: BasicModel, shared_results:ResultsList, env:str, client_id, config, screen=False):
        super(EvaluationProcess, self).__init__()
        self.model = model
        self.env = env
        self.client_id = client_id
        self.shared_results = shared_results
        self.config = config
        self.screen =screen

    def run(self):
        log_debug(f'Evaluator is ready!',self.screen)
        # get connected to one of the ros ports 
        env = start_env(env=self.env,
                speed = self.config.gz_speed,
                client_id =None ,
                max_episode_steps = self.config.max_episode_steps,
                multimodal = self.config.multimodal,
        )
        
        self.model.eval()

        eval_result = fl_evaluate(self.model, env, self.config)
        self.shared_results.value = list(eval_result)






        
def get_shared_params(shared_params,lock):
    try:
        with lock:
            params = pickle.loads(shared_params.value)
        return params
    except Exception as e:
        logging.error(f"Error getting shared params: {e}")
        return None

def log_debug(msg:str,screen:bool):
    if DEBUGGING and screen:
        logger_debug.debug(msg)
