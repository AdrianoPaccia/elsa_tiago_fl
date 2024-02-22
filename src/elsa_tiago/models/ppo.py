import copy
import gym
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from argparse import Namespace
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from collections import OrderedDict
import torch.distributions as distributions
from utils.env_utils import DiscretizedBox

from models.model import BasicModel,MLP,CNN
from utils.communication_utils import log_communication
from utils.logger import Logger
from utils.rl_utils import calculate_returns, calculate_advantages, BasicReplayBuffer, preprocess


class PPO(BasicModel):
    def __init__(
        self, input_dim, action_dim, device, config: Namespace, input_is_image=False,
    ) -> None:
        super(PPO, self).__init__(input_dim, action_dim, device, config, input_is_image)
        self.input_is_image = input_is_image
        match self.input_is_image:
            case True:
                self.actor = CNN(input_dim, action_dim).to(device)
                self.critic = CNN(input_dim, 1).to(device)
            case False:
                self.actor = MLP(input_dim, action_dim).to(device)
                self.critic = MLP(input_dim, 1).to(device)
            

    def fl_train(
        self,
        env: gym.Env,
        optimizer: torch.optim.Optimizer,
        client_id: int,
        logger: Logger,
        config: Namespace,
        replay_buffer: Optional[BasicReplayBuffer] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Trains and returns the updated weights.
        """

        self.train()

        env.reset()
        param_keys = list(self.state_dict().keys())
        parameters = copy.deepcopy(list(self.state_dict().values()))
        keyed_parameters = {n: p.requires_grad for n, p in self.named_parameters()}
        frozen_parameters = [
            not keyed_parameters[n] if n in keyed_parameters else False
            for n, p in self.state_dict().items()
        ]

        pbar = tqdm(total=self.config.num_steps)
        total_critic_loss = []
        total_actor_loss = []

        # Set model weights to state of beginning of federated round
        state_dict = OrderedDict({k: v for k, v in zip(param_keys, parameters)})
        self.load_state_dict(state_dict, strict=True)
        torch.cuda.empty_cache()
        self.train()


        # Perform N provider iterations (each provider has their own dataloader in the non-private case)
        for iter in range(self.config.fl_parameters.iterations_per_fl_round):
            i_step = 0

            while i_step < self.config.num_steps:
                states = []
                actions = []
                log_prob_actions = []
                values = []
                rewards = []
                done = False
                episode_reward = 0.0
                episode_actor_loss = 0.0
                episode_critic_loss = 0.0

                state = env.reset()

                while not done and i_step < self.config.num_steps:

                    #state preprocessing
                    state = preprocess(state,self.input_is_image,self.device)
                    
                    # append state here, not after we get the next state from env.step()
                    states.append(state)

                    #pick the action
                    action_pred, value_pred = self.actor(state), self.critic(state)
                    action_prob = F.softmax(action_pred, dim=-1)
                    dist = distributions.Categorical(action_prob)
                    action = dist.sample()
                    log_prob_action = dist.log_prob(action)

                    state, reward, terminated, truncated, _ = env.step(action.item())
                    reward = torch.FloatTensor([reward]).to(self.device)
                    done = terminated or truncated 

                    actions.append(action.item())
                    log_prob_actions.append(log_prob_action.item())
                    values.append(value_pred)
                    rewards.append(reward)
                    episode_reward += copy.deepcopy(reward).cpu().numpy()

                    i_step += 1
                    pbar.update()

                states = torch.cat(states)
                actions = torch.tensor(actions).to(self.device)
                log_prob_actions = torch.tensor(log_prob_actions).to(self.device)
                #actions = torch.cat(actions)            
                #log_prob_actions = torch.cat(log_prob_actions)
                values = torch.cat(values).squeeze(-1)

                returns = calculate_returns(
                    rewards, self.config.gamma, device=self.device
                )   
                advantages = calculate_advantages(returns, values)

                for _ in range(self.config.ppo_steps):
                    policy_loss, value_loss = self.training_step(
                        states, actions, log_prob_actions, advantages, returns
                    )

                    optimizer.zero_grad()
                    
                    policy_loss.backward()
                    value_loss.backward()

                    optimizer.step()
                    episode_actor_loss += policy_loss.item()
                    episode_critic_loss += value_loss.item()

                episode_actor_loss /= self.config.ppo_steps
                episode_critic_loss /= self.config.ppo_steps
                total_actor_loss.append(episode_actor_loss)
                total_critic_loss.append(episode_critic_loss)

                log_dict = {
                    "Episode actor loss": episode_actor_loss,
                    "Episode critic loss": episode_critic_loss,
                    "Episode reward": episode_reward,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                logger.logger.log(log_dict)
            

        # After all the iterations:
        # Get the update
        agg_update = [
            w - w_0 for w, w_0 in zip(list(self.state_dict().values()), parameters)
        ]  # Get model update
        upd_weights = [
            torch.add(agg_upd, w_0).cpu()
            for agg_upd, w_0, is_frozen in zip(
                agg_update, copy.deepcopy(parameters), frozen_parameters
            )
            if not is_frozen
        ]  # Send weights of NON-Frozen layers.

        pbar.close()

        fl_round_log_dict = {
            "Train/FL Round Actor loss": np.average(total_actor_loss),
            "Train/FL Round Critic loss": np.average(total_critic_loss),
            "fl_round": logger.current_epoch,
        }

        logger.logger.log(fl_round_log_dict)
        log_communication(
            federated_round=config.current_round,
            sender=client_id,
            receiver=-1,
            data=upd_weights,
            log_location=logger.comms_log_file,
        )  # Store only communicated weights (sent parameters).

        # Send the weights to the server
        return upd_weights


    def training_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_prob_actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = advantages.detach()
        log_prob_actions = log_prob_actions.detach()
        actions = actions.detach()
        action_pred = self.actor(states)
        value_pred = self.critic(states)


        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)

        # new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()

        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = (
            torch.clamp(
                policy_ratio,
                min=1.0 - self.config.ppo_clip,
                max=1.0 + self.config.ppo_clip,
            )
            * advantages
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum()
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()

        return policy_loss, value_loss

    def select_action(
        self,
        state: torch.Tensor,
        training: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:   # type: ignore
        if training:
            action_pred = self.actor(state)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            return dist.sample()
        else:
            action_pred = self.actor(state)
            action_prob = F.softmax(action_pred, dim=-1)
            return torch.argmax(action_prob, dim=-1)

