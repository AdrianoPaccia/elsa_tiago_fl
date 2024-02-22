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
from utils.communication_utils import log_communication
from utils.evaluation_utils import fl_evaluate
from utils.logger import Logger
from utils.rl_utils import Transition, BasicReplayBuffer


class DQN(BasicModel):
    def __init__(
        self, input_dim: int, action_dim: int, device, config: Namespace, input_is_image=False
    ) -> None:
        super(DQN, self).__init__(input_dim, action_dim, device, config,input_is_image)
        # input_is_image is the flag to get wherethere the input is an image
        self.input_is_image = input_is_image
        if self.input_is_image:
            self.policy_net = CNN(input_dim, action_dim).to(device)
            self.target_net = CNN(input_dim, action_dim).to(device)
        else:
            self.policy_net = MLP(input_dim, action_dim).to(device)
            self.target_net = MLP(input_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done = 0.0
        

    def update_step(self) -> None:
        self.steps_done += 1.0

    def training_step(self, batch: Transition, config: Namespace) -> torch.TensorType:
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(state_batch.shape[0], device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * config.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        return loss

    def select_action(
        self,
        state: torch.Tensor,
        training: bool = False,
        config: Optional[Namespace] = None,
        action_space: Optional[Space] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:   # type: ignore
        eps = (
            (
                config.eps_end
                + (config.eps_start - config.eps_end)
                * math.exp(-1.0 * self.steps_done / config.eps_decay)
            )
            if training
            else 0.0
        )

        if random.random() >= eps:
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[action_space.sample()]], device=self.device, dtype=torch.long
            )

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

        pbar = tqdm(total=config.num_steps)

        total_loss = 0.0
        total_training_steps = 0

        # Set model weights to state of beginning of federated round
        state_dict = OrderedDict({k: v for k, v in zip(param_keys, parameters)})
        self.load_state_dict(state_dict, strict=True)
        self.train()

        # Perform N provider iterations (each provider has their own dataloader in the non-private case)
        for iter in range(config.fl_parameters.iterations_per_fl_round):
            i_step = 0
            while i_step < config.num_steps:
                episode_loss = 0.0
                state = env.reset()
                state = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                done = False
                while not done:
                    action = self.select_action(
                        state,
                        config=config,
                        training=True,
                        action_space=env.action_space,
                    )
                    observation, reward, terminated, truncated, _ = env.step(
                        action.item()
                    )
                    reward = torch.tensor([reward], device=self.device)
                    done = terminated or truncated

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(
                            observation, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)

                    # Store the transition in the replay_buffer
                    if replay_buffer is not None:
                        replay_buffer.push(state, action, next_state, reward)
                    self.steps_done += 1

                    # Perform one step of the optimization (on the policy network)
                    if len(replay_buffer) > config.min_len_replay_buffer:
                        total_training_steps += 1
                        transitions = replay_buffer.sample(config.batch_size)
                        batch = Transition(*zip(*transitions))

                        # Training step (model-specific)
                        loss = self.training_step(batch=batch, config=config)
                        loss.backward()

                        # Post-loss-backwards function (model-specific)
                        self.post_loss_backwards()
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += loss.item()
                        episode_loss += loss.item()

                        # Post-optimizer step (model-specific)
                        self.post_optimizer_step()

                    # Update environment and number of steps
                    state = next_state
                    if done or i_step == config.num_steps:
                        break
                    i_step += 1
                    pbar.update()

                    # Evaluate during training every training_step_evaluate steps
                    if i_step % config.training_step_evaluate == 0:
                        print("Evaluating....")
                        (
                            avg_reward,
                            std_reward,
                            avg_episode_length,
                            std_episode_length,
                        ) = fl_evaluate(self, env, config)
                        log_dict = {
                            "Avg Reward (during training) ": avg_reward,
                            "Std Reward (during training) ": std_reward,
                        }
                        logger.logger.log(log_dict)
                        break

                log_dict = {
                    "Episode training loss": episode_loss,
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
            "Train/FL Round loss": total_loss / (total_training_steps + 1.0),
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
