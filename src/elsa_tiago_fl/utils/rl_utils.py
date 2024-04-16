import random
import torch
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import copy
import numpy as np
import time


class Transition:
    def __init__(self, state, action, next_state, reward, done) -> None:
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done
        self.name = 'transition'

    def to(self, device='cuda'):
        if self.state is not None:
            self.state = self.state.to(device)
        if self.action is not None:
            self.action = self.action.to(device)
        if self.next_state is not None:
            self.next_state = self.next_state.to(device)
        if self.reward is not None:
            self.reward = self.reward.to(device)
        if self.done is not None:
            self.done = self.reward.to(device)
            

    def to_cpu(self):
        if self.state is not None and self.state.device.type == 'cuda':
            self.state = self.state.cpu()
        if self.action is not None and self.action.device.type == 'cuda':
            self.action = self.action.cpu()
        if self.next_state is not None and self.next_state.device.type == 'cuda':
            self.next_state = self.next_state.cpu()
        if self.reward is not None and self.reward.device.type == 'cuda':
            self.reward = self.reward.cpu()
        if self.done is not None and self.done.device.type == 'cuda':
            self.done = self.reward.cpu()

    def in_cuda(self):
        attributes = [self.state,self.action,self.next_state,self.reward,self.done]
        for att in attributes:
            if torch.is_tensor and not att.is_cuda:
                return False
        return True


def transition_from_batch(batch):
    """
    Convert a batch of transitions in a transition-batch
    """
    s = []
    a = []
    ns = []
    r = []
    d = []
    for t in batch:
        s.append(t.state)
        a.append(t.action)
        ns.append(t.next_state)
        r.append(t.reward)
        d.append(t.done)
    return Transition(s,a,ns,r,d)


class BasicReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)
    
    def get_capacity(self):
        return len(self.memory)/self.capacity

    def push(self,transition:Transition):
        self.memory.append(transition)

    def sample(self, batch_size: int, device='cpu'):
        sample = random.sample(self.memory, batch_size)
        return sample

def get_buffer_variance(points):
    """
    Conpute the normalized trace of the covariance matrix on the overall list of points
    """
    n = len(points)
    mean_vector = np.mean(points, axis=0)
    centered = points - mean_vector
    covariance_matrix = np.dot(centered.T, centered) / (n - 1)
    trace = np.trace(covariance_matrix)
    trace_norm = trace/n
    return trace_norm
    
class TrajectoryHolder(object):
    def __init__(self, input_dim,action_dim,T_horizon,device) -> None:
        self.s_hoder = np.zeros((T_horizon, input_dim),dtype=np.float32)
        self.a_hoder = np.zeros((T_horizon, action_dim),dtype=np.float32) 
        self.r_hoder = np.zeros((T_horizon, 1),dtype=np.float32) 
        self.s_next_hoder = np.zeros((T_horizon, input_dim),dtype=np.float32) 
        self.logprob_a_hoder = np.zeros((T_horizon, action_dim),dtype=np.float32) 
        self.done_hoder = np.zeros((T_horizon, 1),dtype=np.bool_) 
        self.dw_hoder = np.zeros((T_horizon, 1),dtype=np.bool_) 
        self.device=device

    def store(self, s, a, r, s_next, logprob_a, done, dw, idx):
        self.s_hoder[idx] = s
        self.a_hoder[idx] = a
        self.r_hoder[idx] = r
        self.s_next_hoder[idx] = s_next
        self.logprob_a_hoder[idx] = logprob_a
        self.done_hoder[idx] = done
        self.dw_hoder[idx] = dw

    def get_data(self):
        s = torch.from_numpy(self.s_hoder).to(self.device)
        a = torch.from_numpy(self.a_hoder).to(self.device)
        r = torch.from_numpy(self.r_hoder).to(self.device)
        s_next = torch.from_numpy(self.s_next_hoder).to(self.device)
        logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.device)
        done = torch.from_numpy(self.done_hoder).to(self.device)
        dw = torch.from_numpy(self.dw_hoder).to(self.device)
        return s, a, r, s_next, logprob_a, done, dw


class TrajectoryBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self,trajectory:TrajectoryHolder) -> None:
        """Save a trajectory"""
        self.memory.append(trajectory)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)


def calculate_returns(rewards,discount_factor,normalize=False, device="cpu"):
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + discount_factor * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns).to(device)
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize returns
        return returns


def calculate_advantages(returns, values, normalize=False):
    advantages = returns - values

    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages


from PIL import Image
from torchvision import transforms


def preprocess(state:dict, multimodal: bool=False,device='cpu'):
    if multimodal:
        img,g_pos = state
        # process the image
        #img_tsr = torch.torch(copy.deepcopy(img),dtype=float)
        img_tsr = torch.FloatTensor(copy.deepcopy(img))

        img_tsr = img_tsr.permute(2,0,1).unsqueeze(0)
        img_tsr = torch.nn.functional.interpolate(img_tsr,(48,64), mode='bilinear')
        img_tsr = img_tsr.reshape(1,-1)

        #process the pos
        #pos_tsr = torch.tensor(g_pos,dtype=float).unsqueeze(0)
        pos_tsr = torch.FloatTensor(g_pos).unsqueeze(0)
        state_tsr = torch.cat((img_tsr,pos_tsr),dim=1)
    else:
        cubes_dim = 7*5 + 6 + 3
        

        state_flat = []
        i = 0
        cubes = np.ones((7*5))*(-10)
        for cube in state["cubes"].values():
            for k in cube.values():
                cubes[i:i+len(k)] = list(k)
                i+=len(k)
        state_flat.extend(cubes.tolist())

        i = 0
        cylinders = np.ones((6*1))*(-10)
        for cyl in state["cylinders"].values():
            for k in cyl.values():
                cylinders[i:i+len(k)] = list(k)
                i+=len(k)
        state_flat.extend(cylinders.tolist())
        state_flat.extend(state['fk'])


        '''
        for v in state.values
        state_flat = []
        for s in state:
            state_flat.extend(s.flatten())
        #state_tsr = torch.tensor(state_flat,dtype=float).unsqueeze(0)'''
        state_tsr = torch.FloatTensor(state_flat).unsqueeze(0)

    return state_tsr.to(device)

def unprocess(state, state_dim):
    #for multimodal nn to get img and pos from the preprocessed input tsr
    images = []
    poses = []
    b_size, len_tsr = state.shape
    img_dim,pos_dim =state_dim
    split_n = len_tsr - pos_dim
    for i in range(b_size):
        s_i = state[i]
        #get the image
        img_i = s_i[:split_n]
        img_i = img_i.reshape(1,*img_dim)
        images.append(img_i)
        #get the pos
        pos_i = s_i[split_n:]
        poses.append(pos_i.reshape(1,-1))
    return torch.cat(images), torch.cat(poses)

def get_custom_reward(env, split_reward = False):
    """
    REWARD SHAPING: get one of the cubes are get a reward as a linear combination of the distance between:
        - gripper and cube 
        - cube and respective cylinder
    """
    
    gipper_pos = np.array(env.stored_arm_pose[:3])# position of the EE 

    cubes = env.model_state.cubes
    cylinders = env.model_state.cylinders
    #get the Cube Of Interest (the first which is not in the rigth box)
    for i, n in enumerate(env.model_state.cubes_in_cylinders()):
        if n != 1:
            i_COI = i
    try:
        cube_COI_state = list(cubes.values())[i_COI]
        
        cube_pos = np.array(cube_COI_state.position)
        cube_code = cube_COI_state.type_code
        cylinder_pos = np.array(env.model_state.cylinder_of_type(cube_code).position)

        reward_1 = -np.linalg.norm(cube_pos[:2] - cylinder_pos[:2]) #2d distance
        reward_2 = -np.linalg.norm(gipper_pos - cylinder_pos)#3d distance

    except:
        reward_1 = 0
        reward_2 = 0

    reward = 0.5 * reward_1 + 0.5 * reward_2

    if split_reward:
        return reward, reward_1, reward_2
    else:
        return reward


def cheat_action(env):

    verbose = False
    #get the pos of the gripper
    gipper_pos = np.array(env.stored_arm_pose[:3])# position of the EE 

    #get the scene items
    cubes = env.model_state.cubes
    cylinders = env.model_state.cylinders

    '''#get the Cube Of Interest (the first which is not in the rigth box)
    i_COI = None
    for i, n in enumerate(env.model_state.cubes_in_cylinders()):
        if n != 1:
            i_COI = i
    if i_COI is None:
        return [0,0,0,0]
    
    cube_COI = list(cubes.values())[i_COI]
    cube_COI.id = cube_COI.id

    '''

    for v in env.model_state.cubes.values():
        if v.type_code == env.type_target:
            cube_COI = v
            break
    cylinder_COI = env.model_state.cylinders[env.cylinder_target]

    try:
        cube_COI = env.model_state.cubes[env.cube_target]
    except:
        return [0,0,0,0]

    if env.grasped_item == cube_COI.id:
        dist = np.subtract(gipper_pos[:2],cylinder_COI.position[:2])
        is_ontop = np.linalg.norm(dist) < 0.05
        if is_ontop:
            time.sleep(5)
            #leave the cubettto
            action = [0,0,0,1]
            if verbose:print(f"[{cube_COI.id}] leave the cubettto = {action}")
            return action

        else:
            #print(f'{cube_COI.id}: go to that position')
            # go to that position
            proj_pose = cylinder_COI.position
            proj_pose[-1] = 0.5
            action = random_pos_controller(np.array(cube_COI.position), np.array(proj_pose),mod = 1.0)
            if verbose: print(f"[{cube_COI.id}] go to cylinder = {action}")
            return action

    else:
        grasp_item_id,_ = env.get_grasping_obj()
        if grasp_item_id == cube_COI.id:
            # grasp it (is feasible)
            action = [0,0,0,1]
            if verbose: print(f'[{cube_COI.id}] grasp cube - {action}')
            return action
        else:
            # go towards the object
            action = random_pos_controller(np.array(gipper_pos), np.array(cube_COI.position))
            if verbose: print(f'[{cube_COI.id}] go towards the object - {action}')
            return action



def random_pos_controller(pos,t_pos, mod =1.0):
    # genereate an action that drives the gripper towards the target location with a random magnitude
    dist = pos - t_pos - np.array([0.0,0.0,0.2])
    dist_norm = np.array(dist)/abs(max(dist))    
    dist_norm = dist_norm #+ np.array([0.0,0.0,0.2])
    act = [-mod*x for x in dist_norm]
    act.append(random.random() * (-1))
    return np.clip(act,-1.0,1.0).tolist()


