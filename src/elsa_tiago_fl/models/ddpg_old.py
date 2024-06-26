import copy
from argparse import Namespace
from collections import OrderedDict
from typing import Optional, List, Union

import gym
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
import random
from tqdm import tqdm

from models.model import BasicModel
from models.A2C_model import Actor,Critic
from models.utils import hard_update,soft_update,OrnsteinUhlenbeckProcess, log_debug

from elsa_tiago_fl.utils.communication_utils import log_communication
from elsa_tiago_fl.utils.evaluation_utils import fl_evaluate
from elsa_tiago_fl.utils.logger import Logger
from elsa_tiago_fl.utils.rl_utils import Transition, BasicReplayBuffer,transition_from_batch
from elsa_tiago_fl.utils.build_utils import build_optimizer
from collections import namedtuple
import logging
import multiprocessing as mp


DEBUGGING = True
#setup logs managers
#set_logs_level()
if DEBUGGING:
    logging.basicConfig(level=logging.DEBUG)
    logger_debug = mp.get_logger()

EpisodeLoss = namedtuple('EpisodeLoss',['training_loss','policy_loss','value_loss'])

class DDPG(BasicModel):
    def __init__(
        self, input_dim: int, action_dim: int, device, config: Namespace, multimodal=False,
    ) -> None:
        super(DDPG, self).__init__(input_dim, action_dim, device, config, multimodal)

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':config.hidden1, 
            'hidden2':config.hidden2, 
        }
        if self.multimodal:
            pass
        else:
            self.actor = Actor(input_dim, action_dim, **net_cfg).to(device)
            self.actor_target = Actor(input_dim, action_dim, **net_cfg).to(device)

            self.critic = Critic(input_dim, action_dim, **net_cfg)
            self.critic_target = Critic(input_dim, action_dim, **net_cfg)

        # list the parameters for the optimizer        
        self.optimizer_params = list(self.actor.parameters()) + list(self.critic.parameters())

        #update the target network
        hard_update(self.actor_target,self.actor)
        hard_update(self.critic_target,self.critic)

        #Get the noise distribution
        self.noise_distribution = OrnsteinUhlenbeckProcess(
            size = self.actor.action_dim,#continuos_action_dim,
            theta=config.oup_theta,
            mu=config.oup_mu,
            sigma=config.oup_sigma,
            decay = config.oup_decay
        )
        self.steps_done = 0


    # Main methods (train, select_action, training_step)
    #---------------------------------------------------


    def select_action(
        self,
        state: torch.Tensor,
        training: bool = False,
        config: Optional[Namespace] = None,
        action_space: Optional[Space] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  
    
        with torch.no_grad():
            opt_action = self.actor(state).view(1,-1).float()
            if training: 
                noise = self.noise_distribution.sample(self.steps_done)
                noisy_action = torch.clip(opt_action.cpu() + torch.tensor(noise),-1.,1.)
                '''#get the optimal action from the actor
                continuos_action,bool_action = torch.split(opt_action,self.actor.continuos_action_dim,dim=-1)

                #add noise to the continuos part
                noise = self.noise_distribution.sample(self.steps_done)

                noisy_continuos_action = torch.clip(continuos_action.cpu() + torch.tensor(noise),-1.,1.)

                # Stochastically sample boolean action during training, with probability eps
                eps = self.eps_exponential_decay()
                if torch.rand(1) < eps:
                    noisy_bool_action = torch.randint(0, 2, (1, 1))
                else:
                    noisy_bool_action = bool_action.cpu()

                action = torch.cat([noisy_continuos_action,noisy_bool_action],axis=-1)'''
                return noisy_action.squeeze()
                
            else:
                return opt_action.cpu().squeeze()

    
    def training_step(self,batch):
        """
        This function trains from a batch and outputs the loss
        """
        #convert a batch of transitions in a transition-batch
        batch = transition_from_batch(batch)

        policy_loss, value_loss = self.compute_loss(batch)

        # Optimize actor
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        #optimize critic
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        
        '''self.optimizer.zero_grad()
        comb_loss = policy_loss + value_loss 
        comb_loss.backward()
        self.optimizer.step()'''

        # Target update
        soft_update(self.actor,self.actor_target , self.config.tau)
        soft_update(self.critic, self.critic_target, self.config.tau)

        tot_loss = (policy_loss + value_loss).item()

        self.total_loss += tot_loss
        self.episode_loss[0]+= tot_loss
        self.episode_loss[1]+= policy_loss.item()
        self.episode_loss[2]+= value_loss.item()
        self.steps_done += 1

        return tot_loss
        
    
    def compute_loss(self, batch: Transition) -> torch.TensorType:
        # Get the batches
        state_batch = torch.cat(batch.state) #(b,st)
        action_batch = torch.cat(batch.action) #(b,act)
        reward_batch = torch.cat(batch.reward).unsqueeze(-1) #(b,1)
        next_state_batch = torch.cat(batch.next_state) #(b,st)
        terminal_batch = torch.cat(batch.done).unsqueeze(-1) #(b,1)


        with torch.no_grad():
            opt_target_action_batch = self.actor_target(next_state_batch) # (b,act)
            next_q_values = self.critic_target([next_state_batch,opt_target_action_batch]) # (b,1)
            target_q_batch = reward_batch + self.config.gamma * (torch.ones(terminal_batch.shape).cuda() - terminal_batch) * next_q_values
            target_q_batch = target_q_batch.flatten()

        ## Actor loss
        opt_action_batch = self.actor(state_batch)
        policy_loss = -self.critic_target([state_batch,opt_action_batch]).mean()

        ## Critic loss
        q_batch = self.critic([state_batch, action_batch]).flatten()
        criterion = nn.MSELoss()
        value_loss = criterion(q_batch, target_q_batch.detach())


        log_debug(f'q_batch: {q_batch.mean()}',True)
        log_debug(f'target_q_batch: {target_q_batch.mean()}',True)
        return policy_loss, value_loss
        

    # Useful methods
    #------------------------------

    def log_recap(self,what:str,logger:Logger):
        """
        Log the results of the training session into wandb
        """
        if what == 'episode':
            log_dict = {
                "Episode training loss": self.episode_loss[0],
                "Episode policy loss": self.episode_loss[1],
                "Episode value loss": self.episode_loss[2],
                "lr": self.optimizer.param_groups[0]["lr"],
                "epsilon": self.eps_exponential_decay(),
            }
            self.episode_loss = [0.0,0.0,0.0]
        else:
            log_dict = {
            "Train/FL Round loss": self.total_loss / (self.steps_done + 1),
            "fl_round": logger.current_epoch,
            }
        logger.logger.log(log_dict)
        return log_dict

    def eps_exponential_decay(self):
        return self.config.eps_end + (self.config.eps_start - self.config.eps_end) * math.exp(-1.0 * self.steps_done / self.config.eps_decay)

    def loss_coef_exponential_decay(self,):
        return self.config.eps_end + (self.config.eps_start - self.config.eps_end) * math.exp(-1.0 * self.steps_done / self.config.eps_decay)

    def get_executable_action(self, act):
        logging.debug('action original: ',act)
        action = copy.deepcopy(act.numpy())
        action[-1] = action[-1]>0.5
        logging.debug('action to execute: ',action)
        return action

