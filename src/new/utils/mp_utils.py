import multiprocessing as mp
import time
import gym
import pickle
import copy
from models.model import BasicModel
from elsa_tiago_fl.utils.rl_utils import Transition,preprocess
import logging
import torch
from tqdm import tqdm
from elsa_tiago_fl.utils.evaluation_utils import fl_evaluate, Evaluator
from elsa_tiago_fl.utils.utils import tic,toc
import os
import rospy
import rospkg
from elsa_tiago_gym.utils import setup_env
logger = mp.log_to_stderr()
logger.setLevel(logging.DEBUG)


class PolicyUpdateProcess(mp.Process):
    """
    This is the process that every 10sec updates the model wieights getting
        the experience from the replay buffer 
    """
    #def __init__(self, model, lock, logger, optimizer, shared_params, batch_queue, config, env, client_id, termination_event, screen=False):
    def __init__(self, model, lock, logger, optimizer, shared_params, replay_buffer, replay_queues, config, env, client_id, termination_event, screen=False):
        super(PolicyUpdateProcess, self).__init__()
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

    def run(self):
        log_debug('Policy updater ready to update polices!',self.screen)

        # setup the eval environment
        setup_env(self.env)
        rospy.init_node('parallelSimulationNode')
        self.env = gym.make(id=self.env,env_code=self.client_id,max_episode_steps=100)

        self.model.train()
        pbar = tqdm(total=self.config.train_steps)

        # prefilling of the buffer at first
        log_debug(f'Filling replay buffer ...',self.screen)
        tic()
        while len(self.replay_buffer.memory) < self.config.min_len_replay_buffer:
            self.get_transitions()
            perc_capacity = round(self.replay_buffer.get_capacity()*100,0)
        n = self.config.min_len_replay_buffer
        t = round(toc(),3)
        log_debug(f'Filling the {n} exp required {t}sec, giving {n/t} transition/sec',True)

        for iter in range(self.config.fl_parameters.iterations_per_fl_round):
            for i_step in range(self.config.train_steps):
                #tic()

                # get available trasitions
                self.get_transitions()
                batch = self.get_batch()

                # make 1 training iteration and update the shared weigths
                loss = self.model.training_iter(batch)
                self.send_shared_params(self.model.state_dict())

                pbar.update()
                #when is time, do the evaluation 
                if i_step % self.config.training_step_evaluate == 0:
                    logger.debug("Evaluating....")
                    (
                        avg_reward,
                        std_reward,
                        avg_episode_length,
                        std_episode_length,
                    ) = fl_evaluate(self.model, self.env, self.config)
                    log_dict = {
                        "Avg Reward (during training) ": avg_reward,
                        "Std Reward (during training) ": std_reward,
                    }
                    self.logger.logger.log(log_dict)

                #log_debug(f"Training step {i_step}-{iter} - Loss = {loss} - time = {round(toc(),3)}s",True)

            # log the episode loss
            self.model.log_recap('episode',self.logger)
            log_debug(f"END replay buffer - Capacity at {round(self.replay_buffer.get_capacity()*100,0)}%",True)

        #log the avg round loss
        self.model.log_recap('round',self.logger)

        pbar.close()

        self.termination_event.set()

    def send_shared_params(self,state_dict):
        try:
            state_dict_copy = copy.deepcopy(state_dict)
            new_params_bytes = pickle.dumps(state_dict_copy)
            with self.lock:
                self.shared_params.value = new_params_bytes
        except Exception as e:
            logger.debug(f"Error getting shared params: {e}")
            return None
        
    def get_batch(self):
        try:
            #batch = self.batch_queue.get()
            batch = self.replay_buffer.sample(self.config.batch_size)
            for b in batch:
                b.to(self.model.device)
            return batch
        except Exception as e:
            logger.debug(f"Error getting shared params: {e}")
            return None

    def get_transitions(self):
        try:
            for queue in self.replay_queues:
                experiences = []
                if not queue.empty():
                    experiences.extend(queue.rollout())
            for exp in experiences:
                self.replay_buffer.push(exp) 

        except Exception as e:
            logger.debug(f"Error getting shared params: {e}")
            return None


class ExperienceQueue:
    def __init__(self,queue,lock):
        self.queue = queue
        self.lock = lock
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

        
class WorkerProcess(mp.Process):
    """
    This process collects the experience from the environment using the latest model parameters
    """
    def __init__(self, worker_id, model: BasicModel, replay_queue, shared_params, lock, env,env_config, client_id,config, termination_event,screen=False):
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
        self.env_config = env_config


    def run(self):
        log_debug('Worker #{:} ready to get experiences!'.format(self.worker_id),self.screen)

        # get connected to one of the ros ports 
        port = "http://localhost:1135" + str(self.worker_id) + '/'
        setup_env(self.env,port)
        rospy.init_node('parallelSimulationNode')
        self.env = gym.make(id=self.env,
                            env_code=self.worker_id,
                            max_episode_steps=self.env_config['max_episode_steps'],
                            discrete = self.env_config['discrete'],
                            multimodal = self.env_config['multimodal']
                            )
        #self.model.eval()
        tot_step = 0

        while not self.termination_event.is_set():# and not rospy.is_shutdown():
            # get the new params of the model
            state_dict = get_shared_params(self.shared_params,self.lock)
            if state_dict is not None:
                self.model.load_state_dict(state_dict)
            else:
                log_debug('Worker {:} could NOT load new parameters'.format(self.worker_id),self.screen)
            self.model.eval()

            # Explore the environment using the current policy
            observation = self.env.reset()
            state = preprocess(observation,self.model.multimodal,self.model.img_size,self.model.device)
            i_step = 0
            done = False
            while not done and i_step < self.config.num_steps:
                action = self.model.select_action(
                        state,
                        config=self.config,
                        training=True,
                        action_space=self.env.action_space,
                    )
                if self.env_config['discrete']:
                    observation, reward, terminated, _= self.env.step(self.model.executable_act[action.item()]) 
                else:
                    observation, reward, terminated, _= self.env.step(action.item())    

                reward = torch.tensor([reward], device=self.model.device)
                done = terminated #or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = preprocess(observation,self.model.multimodal,self.model.img_size,self.model.device)
                    #next_state = torch.tensor(
                    #    observation, dtype=torch.float32, device=self.model.device
                    #).unsqueeze(0)

                # Put the transition in the queue 
                self.send_transition(state,action,next_state,reward)
 
                state = next_state
                i_step += 1
                tot_step +=1

    def send_transition(self,state,action,next_state,reward):
        try:
            state_clone = state.clone()
            action_clone = action.clone()
            if not next_state == None:
                next_state_clone = next_state.clone()
            else:
                next_state_clone = None
            reward_clone = reward.clone()
            transition = Transition(state_clone, action_clone, next_state_clone,reward_clone )
            transition.to_cpu()
            self.replay_queue.put(transition)

            log_debug('Worker #{:} sent a transition - queue size = {:}!'.format(self.worker_id,self.replay_queue.size()),self.screen)

        except Exception as e:
            logger.Error(f"Error getting shared params: {e}")
            return None


        
def get_shared_params(shared_params,lock):
    try:
        with lock:
            params = pickle.loads(shared_params.value)
        return params
    except Exception as e:
        logger.Error(f"Error getting shared params: {e}")
        return None

def log_debug(what,screen):
    if screen:
        logger.debug(what)
