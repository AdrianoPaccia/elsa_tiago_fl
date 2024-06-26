from argparse import Namespace
from typing import Optional, List, Union
import torch
import torch.nn as nn
import gym
import copy
from elsa_tiago_fl.utils.logger import Logger
from elsa_tiago_fl.utils.rl_utils import BasicReplayBuffer
import torch.nn.functional as F
from collections import OrderedDict



class BasicModel(nn.Module):
    def __init__(
        self, input_dim, action_dim, device: str, config: Namespace, multimodal: bool
    ) -> None:
        super(BasicModel, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.device = device
        self.config = config
        self.multimodal = multimodal

    def select_action(
        self, state: torch.Tensor, training: bool, **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:   # type: ignore
        NotImplemented

    def fl_train(
        self,
        env: gym.Env,
        optimizer: torch.optim.Optimizer,
        client_id: str,
        logger: Logger,
        config: Namespace,
        replay_buffer: Optional[BasicReplayBuffer] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:   # type: ignore
        NotImplemented

    
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
        self.episode_loss = [0.0,0.0,0.0]
        self.total_loss = 0.0

        self.train()
        return parameters, frozen_parameters








class MLP(nn.Module):
    def __init__(self, input_dim, action_dim) -> None:
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        return self.layer2(x)
    
    
class MLP_continue(nn.Module):
    def __init__(self, input_dim, action_dim,bounds) -> None:
        super(MLP_continue, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, action_dim)
        self.bounds=bounds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.tanh(self.layer1(x))
        out = self.layer2(out)
        return out


class CNN(nn.Module):
    def __init__(self, input_dim, action_dim) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=4, stride=2)
        self.value = nn.Linear(in_features=64, out_features=action_dim)
        self.lin = nn.Linear(in_features=108, out_features=64)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
        h = torch.flatten(h, start_dim=1, end_dim=-1) 
        h = torch.flatten(h, start_dim=1, end_dim=-1)
        h = F.relu(self.lin(h))
        value = F.relu(self.value(h)).reshape(-1)
        return value