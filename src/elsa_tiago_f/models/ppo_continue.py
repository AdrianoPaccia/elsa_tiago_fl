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
from models.model import BasicModel
from utils.communication_utils import log_communication
from utils.logger import Logger
from utils.rl_utils import  BasicReplayBuffer, preprocess, TrajectoryHolder
from models.models_A2C import MultimodalCritic,MultimodalActor, BetaActor, GaussianActor_musigma, GaussianActor_mu, Critic
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import statistics 




class PPO(BasicModel):
    def __init__(
        self, input_dim, action_dim, net_width, device, config: Namespace, multimodal:bool,
    ) -> None:
        super(PPO, self).__init__(input_dim, action_dim, device, config, multimodal)

        self.config = config
        self.multimodal = multimodal
        if self.multimodal:
            self.img_size = input_dim[0]
        else: 
            self.img_size = [None]


        if self.multimodal:
            # Build actor
            self.actor = MultimodalActor(
                state_dim =input_dim,
                action_dim = action_dim,
                net_width=net_width,
            ).to(device)

            # Build Critic
            self.critic = MultimodalCritic(
                state_dim =input_dim,
                action_dim = action_dim,
                net_width=net_width,
            ).to(device)
        else:
            # Build actor
            self.Distribution = config.distribution
            if self.Distribution == 'Beta':
                self.actor = BetaActor(input_dim, action_dim, net_width).to(device)
            elif self.Distribution == 'GS_ms':
                self.actor = GaussianActor_musigma(input_dim, action_dim, net_width).to(device)
            elif self.Distribution == 'GS_m':
                self.actor = GaussianActor_mu(input_dim, action_dim, net_width).to(device)
            else: print('Dist Error')

            # Build Critic
            self.critic = Critic(input_dim, net_width).to(device)

    
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

        self.config = config
        self.optimizer = optimizer

        # Build the trajectory holder
        if self.multimodal:
            img_dim = np.array(config.input_dim[0])
            lin_dim = config.input_dim[1]
            input_dim = img_dim.prod() + lin_dim
            self.buffer = TrajectoryHolder(
                        input_dim=input_dim,
                        action_dim=self.config.action_dim,
                        T_horizon=self.config.T_horizon,
                        device=self.device)
        else:
            self.buffer = TrajectoryHolder(
                        input_dim=self.config.input_dim,
                        action_dim=self.config.action_dim,
                        T_horizon=self.config.T_horizon,
                        device=self.device)

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
        episode_actor_losses=[]
        episode_critic_losses=[]

        # Perform N provider iterations (each provider has their own dataloader in the non-private case)
        for iter in range(self.config.fl_parameters.iterations_per_fl_round):
            traj_lenth, total_steps = 0, 0
            episode_reward = 0.0

            while total_steps < self.config.num_steps:
                done = False

                state = env.reset()
                state = preprocess(state,self.multimodal,self.img_size,self.device)

                while not done and total_steps < self.config.num_steps:

                    # get the action and do a step
                    action, logprob = self.select_action(state, training=True) 
                    next_state, reward, done, _ = env.step(action)
                    #print('--\n Action: {:}\n reward {:}\n done {:} '.format(action,reward,done))

                    # new state preprocessing
                    next_state = preprocess(next_state,self.multimodal,self.img_size,self.device)

                    # store the transition data
                    self.buffer.store(
                        state.cpu(),
                        action,
                        reward,
                        next_state.cpu(),
                        logprob,
                        done,done,
                        idx = traj_lenth)

                    # update vars
                    state = next_state
                    episode_reward += copy.deepcopy(reward)
                    traj_lenth += 1
                    total_steps += 1

                    # do a training step if it is time
                    if traj_lenth % self.config.T_horizon == 0:
                        round_actor_loss, round_critic_loss = self.train_step()
                        episode_actor_losses.append(round_actor_loss)
                        episode_critic_losses.append(round_critic_loss)
                        traj_lenth = 0

                    pbar.update()

            total_actor_loss.append(statistics.mean(episode_actor_losses))
            total_critic_loss.append(statistics.mean(episode_critic_losses))

            log_dict = {
                "Episode actor loss": total_actor_loss[-1],
                "Episode critic loss": total_critic_loss[-1],
                "Episode reward": episode_reward,
                "lr":optimizer.param_groups[0]["lr"]
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
            federated_round=self.config.current_round,
            sender=client_id,
            receiver=-1,
            data=upd_weights,
            log_location=logger.comms_log_file,
        )  # Store only communicated weights (sent parameters).

        # Send the weights to the server
        return upd_weights
    
    
    def train_step(self):   
        self.config.entropy_coef*=self.config.entropy_coef_decay

        #Prepare PyTorch data from the holders
        s, a, r, s_next, logprob_a, done, dw = self.buffer.get_data()

        #Use TD+GAE+LongTrajectory to compute Advantage and TD target
        adv, td_target = self.compute_adv_td(s,r,s_next,done,dw)

        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.config.batch_size))
        
        for _ in range(self.config.ppo_steps):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)
            s, a, td_target, adv, logprob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()
            
            round_actor_loss = 0
            round_critic_loss = 0

            for i in range(optim_iter_num):
                index = slice(i * self.config.batch_size, min((i + 1) * self.config.batch_size, s.shape[0]))
                
                #compute the policy loss
                #distribution = self.actor.get_dist(s[index])
                #dist_entropy = distribution.entropy().sum(1, keepdim=True)
                #logprob_a_now = distribution.log_prob(a[index])

                dist_continuous,dist_bool = self.actor.get_dist(s[index])
                dist_entropy =self.actor.compute_entropy(dist_continuous,dist_bool)
                logprob_a_now = self.actor.compute_log_prob(dist_continuous,dist_bool,a[index])
                ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv[index]
                policy_loss = -torch.min(surr1, surr2) - self.config.entropy_coef * dist_entropy
                policy_loss = policy_loss.mean()
                
                #compute the value loss
                value_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name,param in self.critic.named_parameters():
                    if 'weight' in name:
                        value_loss += param.pow(2).sum() * self.config.l2_reg

                # optimization step
                
                self.optimizer.zero_grad()

                comb_loss = policy_loss * self.config.policy_coef + value_loss * self.config.value_coef
                comb_loss.backward()

                self.optimizer.step()

                round_actor_loss+=policy_loss.item()
                round_critic_loss+=value_loss.item()            

        return round_actor_loss/optim_iter_num, round_actor_loss/optim_iter_num
    

    def compute_adv_td(self,s,r,s_next,done,dw):
        
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            #dw for TD_target and Adv
            deltas = r + self.config.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            #done for GAE
            for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.config.gamma * self.config.lambd * adv[-1] * (~mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
            td_target = adv + vs
            adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps
        return adv,td_target



    def select_action(self, state, training:bool=False):
        with torch.no_grad():
            if training:
                # only used when interact with the env
                #dist = self.actor.get_dist(state)
                #a = dist.sample()
                a,logprob_a = self.actor.stochastic_act(state)
                a = torch.clamp(a, -1, 1).cpu().numpy()[0]
                logprob_a = logprob_a.cpu().numpy().flatten()
                return a, logprob_a # both are in shape (adim, 0)
            else:
                # only used when evaluate the policy.Making the performance more stable
                a = self.actor.deterministic_act(state)
                return a.cpu().numpy()[0], None  # action is in shape (adim, 0)



