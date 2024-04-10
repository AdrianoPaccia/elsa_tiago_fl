import copy
from argparse import Namespace
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
#from models.A2C_model import Actor,Critic
from models.actor_critic import Actor,Critic
from models.utils import hard_update,soft_update,OrnsteinUhlenbeckProcess, log_debug

from elsa_tiago_fl.utils.communication_utils import log_communication
from elsa_tiago_fl.utils.evaluation_utils import fl_evaluate
from elsa_tiago_fl.utils.logger import Logger
from elsa_tiago_fl.utils.rl_utils import Transition, BasicReplayBuffer,transition_from_batch
from elsa_tiago_fl.utils.build_utils import build_optimizer
import logging
import multiprocessing as mp



DEBUGGING = True
#setup logs managers
#set_logs_level()
if DEBUGGING:
    logging.basicConfig(level=logging.DEBUG)
    logger_debug = mp.get_logger()


class DDPG(BasicModel):
    def __init__(
        self, input_dim: int, action_dim: int, device, config: Namespace, multimodal=False,
    ) -> None:
        super(DDPG, self).__init__(input_dim, action_dim, device, config, multimodal)

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':config.hidden1, 
            'hidden2':config.hidden2, 
            'learning_rate':float(config.lr)
        }
        if self.multimodal:
            pass
        else:
            self.actor = Actor(input_dim, action_dim, **net_cfg).to(self.device)
            self.actor_target = Actor(input_dim, action_dim, **net_cfg).to(self.device)

            self.critic = Critic(input_dim, action_dim, **net_cfg).to(self.device)
            self.critic_target = Critic(input_dim, action_dim, **net_cfg).to(self.device)

        # list the parameters for the optimizer        
        self.optimizer_params = list(self.actor.parameters()) + list(self.critic.parameters())

        #update the target network
        hard_update(self.actor_target,self.actor)
        hard_update(self.critic_target,self.critic)

        self.action_bounds = [-1,1]

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
            opt_action = self.actor(state).reshape(-1).float()
            if training: 
                if random.random() < self.eps_linear_decay(): #explore
                    action_norm = np.random.uniform(low=self.action_bounds[0], high=self.action_bounds[1], size=opt_action.shape)
                    action = torch.from_numpy(action_norm).squeeze()
                    return action
                else: #exploit
                    action = opt_action.cpu().squeeze()
                    return action
            else:
                action = opt_action.cpu().squeeze()
                return action

    
    def training_step(self,batch):
        """
        This function trains from a batch and outputs the loss
        """
        states = torch.cat(batch.state) #(b,st)
        actions = torch.cat(batch.action) #(b,act)
        rewards = torch.cat(batch.reward).unsqueeze(-1) #(b,1)
        next_states = torch.cat(batch.next_state) #(b,st)
        dones = torch.cat(batch.done).unsqueeze(-1) #(b,1)

        next_actions = self.actor_target(next_states).detach()
        target_Q = self.critic_target(next_states,next_actions)  
        target_Q = rewards + (self.config.gamma * target_Q * (1 - dones))
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())  # nn.MSELoss() means Mean Squared Error
        self.critic.optimizer.zero_grad()  # .zero_grad() clears old gradients from the last step
        critic_loss.backward()  # .backward() computes the derivative of the loss
        self.critic.optimizer.step()  # .step() is to update the parameters

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()  # .mean() is to calculate the mean of the tensor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Target update
        soft_update(self.actor,self.actor_target , self.config.tau)
        soft_update(self.critic, self.critic_target, self.config.tau)

        tot_loss = (actor_loss + critic_loss).item()
        self.total_loss += tot_loss
        self.episode_loss[0]+= tot_loss
        self.episode_loss[1]+= actor_loss.item()
        self.episode_loss[2]+= critic_loss.item()
        self.steps_done += 1

        return tot_loss




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
                "epsilon": self.eps_linear_decay(),
            }
            self.episode_loss = [0.0,0.0,0.0]
        else:
            log_dict = {
            "Train/FL Round loss": self.total_loss / (self.steps_done + 1),
            "fl_round": logger.current_epoch,
            }
        logger.logger.log(log_dict)
        return log_dict

    def eps_linear_decay(self):
        epsilon = np.interp(self.steps_done, [0, self.config.eps_decay],[self.config.eps_start , self.config.eps_end])  
        return epsilon

    def get_executable_action(self, action):
        return action.tolist()

