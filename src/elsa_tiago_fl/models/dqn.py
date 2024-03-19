import copy
from argparse import Namespace
from collections import OrderedDict
from typing import Optional, List, Union

import gym
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
import random
from tqdm import tqdm

from models.model import BasicModel,MLP,CNN
from models.utils import hard_update

from elsa_tiago_fl.utils.communication_utils import log_communication
from elsa_tiago_fl.utils.evaluation_utils import fl_evaluate
from elsa_tiago_fl.utils.logger import Logger
from elsa_tiago_fl.utils.rl_utils import Transition, BasicReplayBuffer,transition_from_batch


class DQN(BasicModel):
    def __init__(
        self, input_dim: int, action_dim: int, device, config: Namespace,
    ) -> None:
        super(DQN, self).__init__(input_dim, action_dim, device, config, multimodal)

        #discretize the action space
        self.executable_act = self.discrete_action_space(config.action_dim)

        #get the trainable models
        if self.multimodal:
            self.policy_net = CNN(input_dim, action_dim).to(device)
            self.target_net = CNN(input_dim, action_dim).to(device)
        else:
            self.policy_net = MLP(input_dim, action_dim).to(device)
            self.target_net = MLP(input_dim, action_dim).to(device)
            
        # list the parameters for the optimizer        
        self.optimization_params = self.policy_net.parameters()

        #update the target network
        hard_update(self.target_net,self.policy_net)

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
        if training:
            eps = config.eps_end + (config.eps_start - config.eps_end) * math.exp(-1.0 * self.steps_done / config.eps_decay)
            if random.random() >= eps:
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor(
                    [[action_space.sample()]], device=self.device, dtype=torch.long
                ).view(1, 1)
        else:
            return self.policy_net(state).max(1)[1].view(1).item()

    
    def training_step(self,batch):
        """
        This function trains from a batch and outputs the loss
        """
        #convert a batch of transitions in a transition-batch
        batch = transition_from_batch(batch)

        # Training step (model-specific)
        loss = self.compute_loss(batch=batch)
        loss.backward()

        # Post-loss-backwards function (model-specific)
        self.post_loss_backwards()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Post-optimizer step (model-specific)
        self.post_optimizer_step()

        self.total_loss += loss.item()
        self.episode_loss += loss.item()
        self.total_training_steps += 1
        return loss
    
    def compute_loss(self, batch: Transition) -> torch.TensorType:
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        terminal_batch = torch.cat(batch.done)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0]
        expected_state_action_values = reward_batch + (torch.ones(terminal_batch.shape) - terminal_batch) @ (next_state_values * self.config.gamma)  

        criterion = nn.SmoothL1Loss() # Huber loss
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        return loss


    # Useful methods
    #------------------------------
    def post_loss_backwards(self) -> None:
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1000)

    def post_optimizer_step(self) -> None:
        if self.steps_done % self.config.update_target_net == 0:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * self.config.tau + target_net_state_dict[key] * (1 - self.config.tau)
            self.target_net.load_state_dict(target_net_state_dict)

    def setup_fl_training(self,optimizer):
        """
        Setup the model for the Federated Learning training steps
        """
        self.train()
        param_keys = list(self.state_dict().keys())
        parameters = copy.deepcopy(list(self.state_dict().values()))
        keyed_parameters = {n: p.requires_grad for n, p in self.named_parameters()}
        frozen_parameters = [
            not keyed_parameters[n] if n in keyed_parameters else False
            for n, p in self.state_dict().items()
        ]
        # Set model weights to state of beginning of federated round
        state_dict = OrderedDict({k: v for k, v in zip(param_keys, parameters)})
        self.load_state_dict(state_dict, strict=True)
        self.optimizer = optimizer
        self.episode_loss = 0.0
        self.total_loss = 0.0
        self.total_training_steps = 0

        self.train()
        return parameters, frozen_parameters

    def discrete_action_space(self,action_dim):
        """
        Discretize the action space so that each joint value can be in {-1,0,1}
        """
        discrete_act = {i: [0]*action_dim for i in range(action_dim*2-1)}
        for i in range(action_dim-1):
            discrete_act[i*2][i]=1.0
            discrete_act[i*2+1][i]=-1.0
        discrete_act[(action_dim-1)*2][-1] = 1 #grasping action
        return discrete_act

    def log_recap(self,what:str,logger:Logger):
        """
        Log the results of the training session into wandb
        """
        if what == 'episode':
            log_dict = {
                "Episode training loss": self.episode_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.episode_loss = 0
        else:
            log_dict = {
            "Train/FL Round loss": self.total_loss / (self.total_training_steps + 1),
            "fl_round": logger.current_epoch,
            }
        logger.logger.log(log_dict)
        return log_dict

    
    def get_executable_action(self, act):
        return self.executable_act[act]
        
