from tabnanny import check
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from .parameters import Parameters
from . import replay_memory
from .utils import *
import numpy as np
import os


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class GeneticAgent:
    def __init__(self, args: Parameters, checkpoint_folder=None):
        self.args = args
        self.id = self.args.count_actors
        self.args.count_actors += 1
        self.actor = Actor(args)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-3)

        self.buffer = replay_memory.ReplayMemory(self.args.individual_bs, args.device)
        self.loss = nn.MSELoss()
        self.yet_eval = False
        if checkpoint_folder is not None:
            self.load_info(checkpoint_folder)
            self.actor.train()

    def update_parameters(self, batch, p1, p2, critic, scalar_weight):
        state_batch, _, _, _, _ = batch

        # --- 1. Prepare conditioned state for the critic ---
        if self.args.weight_conditioned:
            # Convert weight to a [batch_size, num_objectives] tensor
            weight_tensor = torch.FloatTensor(scalar_weight).to(self.args.device)
            weight_batch = weight_tensor.expand(state_batch.shape[0], -1)
            # Concatenate state and weight
            critic_state_in = torch.cat([state_batch, weight_batch], dim=1)
        else:
            # Use the raw state if not conditioned
            critic_state_in = state_batch

        # --- 2. Get Q-values from critic using the (potentially) conditioned state ---
        # Actors p1 and p2 get the *raw* state
        p1_action = p1(state_batch)
        p2_action = p2(state_batch)
        
        # Critic gets the *conditioned* state
        p1_q = critic(critic_state_in, p1_action).flatten()
        p2_q = critic(critic_state_in, p2_action).flatten()

        # --- 3. Filter actions and states ---
        eps = 0.0
        action_batch = torch.cat((p1_action[p1_q - p2_q > eps], p2_action[p2_q - p1_q >= eps])).detach()
        
        # The state_batch here is filtered, but remains the *raw* state
        state_batch = torch.cat((state_batch[p1_q - p2_q > eps], state_batch[p2_q - p1_q >= eps]))
        
        # The agent's own actor also gets the *raw* state
        actor_action = self.actor(state_batch)

        # --- 4. Actor Update ---
        self.actor_optim.zero_grad()
        sq = (actor_action - action_batch)**2
        policy_loss = torch.sum(sq) + torch.mean(actor_action**2)
        policy_mse = torch.mean(sq)
        policy_loss.backward()
        self.actor_optim.step()

        return policy_mse.item()

    def save_info(self, folder_path):
        checkpoint = os.path.join(folder_path, "state_dicts.pkl")
        torch.save({
            'actor_sd': self.actor.state_dict(),
            'actor_optim_sd': self.actor_optim.state_dict(),
            'id': self.id
            }, checkpoint)
        buffer_path = os.path.join(folder_path, "buffer.npy")
        self.buffer.save_info(buffer_path)
    
    def load_info(self, folder_path):
        checkpoint = os.path.join(folder_path, "state_dicts.pkl")
        buffer_path = os.path.join(folder_path, "buffer.npy")
        self.buffer.load_info(buffer_path)
        checkpoint_sd = torch.load(checkpoint, map_location=self.args.device)
        self.actor.load_state_dict(checkpoint_sd['actor_sd'])
        self.actor_optim.load_state_dict(checkpoint_sd['actor_optim_sd'])
        if 'id' in checkpoint_sd:
            self.id = int(checkpoint_sd['id'])


class Actor(nn.Module):
    def __init__(self, args, init=False):
        super(Actor, self).__init__()
        self.args = args
        l1 = args.ls 
        l2 = args.ls

        # Construct Hidden Layer 1
        self.w_l1 = nn.Linear(args.state_dim, l1)
        if self.args.use_ln: self.lnorm1 = LayerNorm(l1)

        # Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)
        if self.args.use_ln: self.lnorm2 = LayerNorm(l2)

        # Out
        self.w_out = nn.Linear(l2, args.action_dim)

        # Init
        if init:
            self.w_out.weight.data.mul_(0.1)
            self.w_out.bias.data.mul_(0.1)
        
        self.ounoise = OUNoise(args.action_dim)

        self.to(self.args.device)

    def forward(self, input):

        # Hidden Layer 1
        out = self.w_l1(input)
        if self.args.use_ln: out = self.lnorm1(out)
        out = out.tanh()

        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = out.tanh()

        # Out
        out = (self.w_out(out)).tanh()
        return out

    def select_action(self, state, is_ounoise=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        action = self.forward(state).cpu().data.numpy().flatten()
        if is_ounoise:
            action += self.ounoise.noise()
        return action

    # function to return current pytorch gradient in same order as genome's flattened parameter vector
    def extract_grad(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.grad.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to grab current flattened neural network weights
    def extract_parameters(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to inject a flat vector of ANN parameters into the model's current neural network weights
    def inject_parameters(self, pvec):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = pvec[count:count + sz]
            reshaped = raw.view(param.size())
            param.data.copy_(reshaped.data)
            count += sz

    # count how many parameters are in the model
    def count_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            count += param.numel()
        return count


class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        l1 = 200; l2 = 300

        # Construct input interface (Hidden Layer 1)
        # Alter state input size based on if weight conditioning is required or not
        self.w_state_l1 = nn.Linear(args.state_dim+args.num_objectives, l1) if args.weight_conditioned==True else nn.Linear(args.state_dim, l1)
        self.w_action_l1 = nn.Linear(args.action_dim, l1)

        # Hidden Layer 2
        self.w_l2 = nn.Linear(2*l1, l2)
        if self.args.use_ln: self.lnorm2 = LayerNorm(l2)

        # Out
        self.w_out = nn.Linear(l2, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, input, action):

        # Hidden Layer 1 (Input Interface)
        out_state = F.elu(self.w_state_l1(input))
        out_action = F.elu(self.w_action_l1(action))
        out = torch.cat((out_state, out_action), 1)

        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = F.elu(out)

        # Output interface
        out = self.w_out(out)

        return out


class DDPG(object):
    def __init__(self, args, scalar_weight: np.ndarray, other_weights: np.ndarray, checkpoint_folder=None):

        self.args = args
        self.scalar_weight = scalar_weight
        self.other_weights = other_weights
        self.buffer = replay_memory.ReplayMemory(args.buffer_size, args.device)

        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr=0.5e-4)

        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.5e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        if checkpoint_folder is not None:
            self.load_info(checkpoint_folder)
            self.actor.train()
            self.actor_target.train()
            self.critic.train()
            self.critic_target.train()

    def td_error(self, state, action, next_state, reward, done):
        next_action = self.actor_target.forward(next_state)
        next_q = self.critic_target(next_state, next_action)

        done = 1 if done else 0
        if self.args.use_done_mask: next_q = next_q * (1 - done)  # Done mask
        target_q = reward + (self.gamma * next_q)

        current_q = self.critic(state, action)
        dt = (current_q - target_q).abs()
        return dt.item()

    def update_parameters(self, batch):
            # 1. Unpack and move all batch data to device ONCE
            state_batch, action_batch, reward_vec, next_state_batch, done_batch = batch
            
            state_batch = state_batch.to(self.args.device)
            action_batch = action_batch.to(self.args.device)
            reward_vec = reward_vec.to(self.args.device) # This is the reward *vector*
            next_state_batch = next_state_batch.to(self.args.device)
            
            # Handle done mask
            if self.args.use_done_mask:
                done_batch = done_batch.to(self.args.device)
            
            # Initialise the critic loss to return
            delta = torch.tensor(0.0)
            
            # Load networks to GPU (in case they aren't)
            self.actor_target.to(self.args.device)
            self.critic_target.to(self.args.device)
            self.critic.to(self.args.device)

            # 2. Create the list of weights to loop over
            # If not conditioned, this list just has one item: self.scalar_weight
            # If conditioned, it has all weights. This handles both cases.
            weight_vectors = np.concatenate(([self.scalar_weight], self.other_weights), axis=0) if self.args.weight_conditioned else [self.scalar_weight]

            # --- 3. Critic Update Loop (Generalist Critic) ---
            for weight_np in weight_vectors:
                
                # --- 3a. Prepare inputs for this iteration ---
                
                # Convert numpy weight to a
                # [batch_size, num_objectives] tensor
                weight_tensor = torch.FloatTensor(weight_np).to(self.args.device)
                weight_batch = weight_tensor.expand(state_batch.shape[0], -1)

                # Scalarize reward using the ORIGINAL reward_vec
                scalar_reward_batch = torch.matmul(reward_vec, weight_tensor.reshape(-1, 1))

                # --- 3b. Condition states based on args (THE CORE REQUEST) ---
                if self.args.weight_conditioned:
                    state_in = torch.cat([state_batch, weight_batch], dim=1)
                    next_state_in = torch.cat([next_state_batch, weight_batch], dim=1)
                else:
                    # If not conditioned, just use the original states
                    state_in = state_batch
                    next_state_in = next_state_batch

                # --- 3c. Compute Critic Target ---
                
                # Actor *always* gets the raw (non-conditioned) state
                next_action_batch = self.actor_target.forward(next_state_batch) 
                
                with torch.no_grad():
                    # Target Critic gets the (potentially) conditioned state
                    next_q = self.critic_target.forward(next_state_in, next_action_batch)
                    
                    if self.args.use_done_mask: 
                        next_q = next_q * (1 - done_batch) #Done mask
                        
                    target_q = scalar_reward_batch + (self.gamma * next_q)

                # --- 3d. Compute Critic Loss and Update ---
                
                # Critic gets the (potentially) conditioned state
                current_q = self.critic.forward(state_in, action_batch)
                delta = (current_q - target_q).abs()
                dt = torch.mean(delta**2)
                
                self.critic_optim.zero_grad()
                dt.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
                self.critic_optim.step()

            # --- 4. Actor Update (Specialist Actor) ---
            self.actor_optim.zero_grad()

            # --- 4a. Condition state for actor's critic (THE CORE REQUEST) ---
            # We only use the main self.scalar_weight for the actor update
            if self.args.weight_conditioned:
                main_weight_tensor = torch.FloatTensor(self.scalar_weight).to(self.args.device)
                main_weight_batch = main_weight_tensor.expand(state_batch.shape[0], -1)
                state_for_actor_critic = torch.cat([state_batch, main_weight_batch], dim=1)
            else:
                state_for_actor_critic = state_batch
                
            # --- 4b. Compute Actor Loss ---
            
            # Actor *always* gets the raw state
            actor_actions = self.actor.forward(state_batch)
            
            # Critic gets the state conditioned *only* on the main weight
            policy_grad_loss = -(self.critic.forward(state_for_actor_critic, actor_actions)).mean()
            policy_loss = policy_grad_loss

            # --- 4c. Actor Update ---
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_optim.step()

            # --- 5. Soft Update Target Networks ---
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
            
            # 'delta' will be from the last iteration of the critic loop
            return policy_grad_loss.data.cpu().numpy(), delta.data.cpu().numpy()

    def save_info(self, folder_path):
        checkpoint = os.path.join(folder_path, "state_dicts.pkl")
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_t': self.actor_target.state_dict(),
            'actor_op': self.actor_optim.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_t': self.critic_target.state_dict(),
            'critic_op': self.critic_optim.state_dict(),
        }, checkpoint)
        buffer_path = os.path.join(folder_path, "buffer.npy")
        self.buffer.save_info(buffer_path)
        ou_path  = os.path.join(folder_path, "ou.npy")
        with open(ou_path, 'wb') as f:
            np.save(f, self.actor.ounoise.state)

        
    def load_info(self, folder_path):
        checkpoint = os.path.join(folder_path, "state_dicts.pkl")
        checkpoint_sd = torch.load(checkpoint, map_location=self.args.device)
        self.actor.load_state_dict(checkpoint_sd['actor'])
        self.actor_target.load_state_dict(checkpoint_sd['actor_t'])
        self.actor_optim.load_state_dict(checkpoint_sd['actor_op'])
        self.critic.load_state_dict(checkpoint_sd['critic'])
        self.critic_target.load_state_dict(checkpoint_sd['critic_t'])
        self.critic_optim.load_state_dict(checkpoint_sd['critic_op'])
        buffer_path = os.path.join(folder_path, "buffer.npy")
        self.buffer.load_info(buffer_path)
        ou_path  = os.path.join(folder_path, "ou.npy")
        with open(ou_path, 'rb') as f:
            ou_state = np.load(f)
            self.actor.ounoise.state = ou_state

def fanin_init(size, fanin=None):
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)

def actfn_none(inp): return inp

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class OUNoise:

    def __init__(self, action_dimension, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
