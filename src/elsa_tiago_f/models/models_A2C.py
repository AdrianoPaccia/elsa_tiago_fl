import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal, Bernoulli
from utils.rl_utils import unprocess



class BetaActor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(BetaActor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.alpha_head = nn.Linear(net_width, action_dim)
        self.beta_head = nn.Linear(net_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))

        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0

        return alpha,beta

    def get_dist(self,state):
        alpha,beta = self.forward(state)
        dist = Beta(alpha, beta)
        
        return dist

    def deterministic_act(self, state):
        alpha, beta = self.forward(state)
        mode = (alpha) / (alpha + beta)
        return mode

class GaussianActor_musigma(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(GaussianActor_musigma, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.sigma_head = nn.Linear(net_width, action_dim)

        # output layers 
        self.mu_continuous_head = nn.Linear(net_width, out_features=3)
        self.sigma_continuous_head = nn.Linear(net_width, out_features=3)
        self.bool_head = nn.Linear(net_width, out_features=1)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mu_continuous = torch.tanh(self.mu_continuous_head(x))
        sigma_continuous = F.softplus( self.sigma_continuous_head(x) )
        bool_prob = torch.sigmoid(self.bool_head(x))
        return mu_continuous,sigma_continuous,bool_prob

    def get_dist(self, state):
        mu_continuous,sigma_continuous,bool_prob = self.forward(state)
        # continuous distribution
        dist_continuous = Normal(mu_continuous,sigma_continuous)
        # Bool (Bernoulli) distribution
        dist_bool = Bernoulli(bool_prob)
        return dist_continuous,dist_bool

    def deterministic_act(self, state):
        mu, _,bool_prob = self.forward(state)
        return torch.cat([mu, (bool_prob >= 0.5).float()], dim=-1)
        
    def stochastic_act(self,state):
        continuous_dist,bool_dist = self.get_dist(state)
        a_continuous = continuous_dist.sample()
        a_bool = bool_dist.sample()
        # compute the og prob
        log_prob_continuous = continuous_dist.log_prob(a_continuous)
        log_prob_boolean = bool_dist.log_prob(a_bool)
        log_prob = torch.cat((log_prob_continuous,log_prob_boolean),dim=-1)
        return torch.cat([a_continuous, a_bool], dim=-1), log_prob
    
    def compute_log_prob(self,continuous_dist,bool_dist,action):
        a_continuous, a_bool = torch.split(action, [3,1], dim=-1)
        log_prob_continuous = continuous_dist.log_prob(a_continuous)
        log_prob_boolean = bool_dist.log_prob(a_bool)
        return torch.cat((log_prob_continuous,log_prob_boolean),dim=-1)

    def compute_entropy(self,continuous_dist,bool_dist):
        # Calculate entropy for both components
        entropy_continuous = continuous_dist.entropy().sum(1, keepdim=True)
        entropy_boolean = bool_dist.entropy().sum(1, keepdim=True)
        return entropy_continuous + entropy_boolean


    





class GaussianActor_mu(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, log_std=0):
        super(GaussianActor_mu, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.mu_head.weight.data.mul_(0.1)
        self.mu_head.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        return mu

    def get_dist(self,state):
        mu = self.forward(state)
        action_log_std = self.action_log_std.expand_as(mu)
        action_std = torch.exp(action_log_std)

        dist = Normal(mu, action_std)
        return dist

    def deterministic_act(self, state):
        return self.forward(state)


class Critic(nn.Module):
    def __init__(self, state_dim,net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.tanh(self.C1(state))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v


class MultimodalActor(nn.Module):
    def __init__(self, state_dim:list, action_dim:int,net_width: list):
        super(MultimodalActor, self).__init__()

        self.img_dim, self.lin_dim = state_dim
        n_channels = self.img_dim[0]
        net_width_1, net_width_2, net_width_3 = net_width
        k_1 = 1 # 32
        k_2 = 1 #64
        # image processing layers
        self.img_conv1 = nn.Conv2d(n_channels, k_1, kernel_size=3, stride=1, padding=1)
        self.img_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img_conv2 = nn.Conv2d(k_1, k_2, kernel_size=3, stride=1, padding=1)
        self.img_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img_fc1 = nn.Linear(768, net_width_1)
        

        # positional data processing layers
        self.pos_fc1 = nn.Linear(3, net_width_2)

        # combination layer
        self.fc_comb = nn.Linear(net_width_1 + net_width_2, net_width_3)

        # output layers 
        self.mu_continuous_head = nn.Linear(in_features=net_width_3, out_features=3)
        self.sigma_continuous_head = nn.Linear(in_features=net_width_3, out_features=3)
        self.sigma_bool_head = nn.Linear(in_features=net_width_3, out_features=1)


    def forward(self, obs):
        img, pos = unprocess(obs,(self.img_dim, self.lin_dim))

        # process image
        img = F.tanh(self.img_conv1(img))
        img = self.img_pool1(img)
        img = F.tanh(self.img_conv2(img))
        img = self.img_pool2(img)
        img = img.view(-1,768)#  64 * 120 * 160)  # Flatten the img tensor
        img = F.tanh(self.img_fc1(img))
        
		# process positions
        pos = F.tanh(self.pos_fc1(pos))
        
        #combination 
        comb = torch.cat((img, pos), dim=1)
        comb = F.tanh(self.fc_comb(comb))
        
		#get sigma and mu
        mu_continuous = torch.sigmoid(self.mu_continuous_head(comb))
        sigma_continuous = F.softplus( self.sigma_bool_head(comb) )
        mu_bool = torch.sigmoid(self.sigma_bool_head)

        return mu_continuous,sigma_continuous,mu_bool

    def get_dist(self, state):
        mu,sigma = self.forward(state)
        dist = Normal(mu,sigma)
        return dist

    def deterministic_act(self, state):
        mu, sigma = self.forward(state)
        return mu



# critic takes in input images and positions 
class MultimodalCritic(nn.Module):
    def __init__(self,state_dim:list, action_dim:int,net_width: list):
        super(MultimodalCritic, self).__init__()
        self.img_dim, self.lin_dim = state_dim
        n_channels = self.img_dim[0]
        net_width_1, net_width_2, net_width_3 = net_width
        k_1 = 1 # 32
        k_2 = 1 #64

        # image processing layers
        self.img_conv1 = nn.Conv2d(n_channels, k_1, kernel_size=3, stride=1, padding=1)
        self.img_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img_conv2 = nn.Conv2d(k_1, k_2, kernel_size=3, stride=1, padding=1)
        self.img_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img_fc1 = nn.Linear(768, net_width_1)

        # positional data processing layers
        self.pos_fc1 = nn.Linear(3, net_width_2)

        # combination layer
        self.fc_comb = nn.Linear(net_width_1 + net_width_2, net_width_3)

        # output layer
        self.fc_output = nn.Linear(net_width_3, 1)

    def forward(self, obs):
        img, pos = unprocess(obs,(self.img_dim, self.lin_dim))
       
        img = F.tanh(self.img_conv1(img))
        img = self.img_pool1(img)
        img = F.tanh(self.img_conv2(img))
        img = self.img_pool2(img)
        img = img.view(-1, 768)  # Flatten the img tensor
        img = F.tanh(self.img_fc1(img))

        pos = F.tanh(self.pos_fc1(pos))
        
        #combination 
        comb = torch.cat((img, pos), dim=1)
        comb = F.tanh(self.fc_comb(comb))

        out = self.fc_output(comb)

        return out



'''
        self.conv1 = nn.Conv2d(in_channels=img_dim[2], out_channels=9, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=4, stride=2)
        self.lin = nn.Linear(in_features=108, out_features=net_width)
        self.value = nn.Linear(in_features=net_width, out_features=1)

    def forward(self, obs):
        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
        h = torch.flatten(h, start_dim=1, end_dim=-1) 
        h = F.relu(self.lin(h))
        value = self.value(h).reshape(-1)

        v = torch.tanh(self.C1(state))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v
'''

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise


def Action_adapter(a,max_action):
    #from [0,1] to [-max,max]
    return  2*(a-0.5)*max_action

def Reward_adapter(r, EnvIdex):
    # For BipedalWalker
    if EnvIdex == 0 or EnvIdex == 1:
        if r <= -100: r = -1
    # For Pendulum-v0
    elif EnvIdex == 3:
        r = (r + 8) / 8
    return r

def evaluate_policy(env, agent, max_action, turns):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            a, logprob_a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
            act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
            s_next, r, dw, tr, info = env.step(act)
            done = (dw or tr)

            total_scores += r
            s = s_next

    return total_scores/turns