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

        # --- Modified Critic Initialization ---
        if self.args.multi_critics:
            # Create a separate critic for each objective (assuming canonical islands e.g. [1,0], [0,1])
            self.critics = nn.ModuleList([Critic(args) for _ in range(args.num_objectives)])
            self.critics_target = nn.ModuleList([Critic(args) for _ in range(args.num_objectives)])
            self.critics_optims = [Adam(c.parameters(), lr=0.5e-3) for c in self.critics]
            
            # Sync targets
            for c, c_t in zip(self.critics, self.critics_target):
                hard_update(c_t, c)
        else:
            # Original Single Critic
            self.critic = Critic(args)
            self.critic_target = Critic(args)
            self.critic_optim = Adam(self.critic.parameters(), lr=0.5e-3)
            hard_update(self.critic_target, self.critic)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        if checkpoint_folder is not None:
            self.load_info(checkpoint_folder)
            self.actor.train()
            self.actor_target.train()
            if self.args.multi_critics:
                for c, ct in zip(self.critics, self.critics_target):
                    c.train(); ct.train()
            else:
                self.critic.train()
                self.critic_target.train()

    def update_parameters(self, batch):
        # Unpack batch and move components to GPU/Device
        state_batch, action_batch, reward_vec, next_state_batch, done_batch = batch
        
        state_batch = state_batch.to(self.args.device)
        action_batch = action_batch.to(self.args.device)
        reward_vec = reward_vec.to(self.args.device) 
        next_state_batch = next_state_batch.to(self.args.device)
        
        if self.args.use_done_mask:
            done_batch = done_batch.to(self.args.device)
        
        delta = torch.tensor(0.0)
        
        # Ensure target networks are on the correct device before computation
        self.actor_target.to(self.args.device)
        if self.args.multi_critics:
            for c in self.critics: c.to(self.args.device)
            for c_t in self.critics_target: c_t.to(self.args.device)
        else:
            self.critic.to(self.args.device); self.critic_target.to(self.args.device)

        # Prepare list of weights to train on. 
        # Includes the actor's primary weight + auxiliary weights (other islands) if conditioning is enabled.
        weight_vectors = np.concatenate(([self.scalar_weight], self.other_weights), axis=0) if self.args.weight_conditioned else [self.scalar_weight]

        # --- Critic Update Loop ---
        # We iterate through all relevant weight vectors to update the corresponding critic (or the single shared critic).
        for weight_np in weight_vectors:
            
            # 1. Identify which critic network to use for this specific weight vector
            if self.args.multi_critics:
                # Map one-hot weight (e.g., [1,0]) to the corresponding index (0) to select the specialist critic
                obj_idx = np.argmax(weight_np) 
                curr_critic = self.critics[obj_idx]
                curr_target_critic = self.critics_target[obj_idx]
                curr_optim = self.critics_optims[obj_idx]
            else:
                # Fallback to the single generalist critic
                curr_critic = self.critic
                curr_target_critic = self.critic_target
                curr_optim = self.critic_optim

            # 2. Prepare Data: Scalarize rewards and Condition States
            weight_tensor = torch.FloatTensor(weight_np).to(self.args.device)
            weight_batch = weight_tensor.expand(state_batch.shape[0], -1)
            
            # Compute scalar reward: Dot product of reward vector and current weight vector
            scalar_reward_batch = torch.matmul(reward_vec, weight_tensor.reshape(-1, 1))

            # Concatenate weight to state if 'weight_conditioned' is True (UVFA style)
            if self.args.weight_conditioned:
                state_in = torch.cat([state_batch, weight_batch], dim=1)
                next_state_in = torch.cat([next_state_batch, weight_batch], dim=1)
            else:
                state_in = state_batch
                next_state_in = next_state_batch

            # 3. Compute Bellman Target
            # Note: Actor target always receives the raw state, regardless of conditioning
            next_action_batch = self.actor_target.forward(next_state_batch)
            
            with torch.no_grad():
                # Target critic evaluates (State + Weight, Next Action)
                next_q = curr_target_critic.forward(next_state_in, next_action_batch)
                if self.args.use_done_mask: 
                    next_q = next_q * (1 - done_batch)
                target_q = scalar_reward_batch + (self.gamma * next_q)

            # 4. Update Critic
            # Current critic evaluates (State + Weight, Current Action)
            current_q = curr_critic.forward(state_in, action_batch)
            delta = (current_q - target_q).abs()
            dt = torch.mean(delta**2)
            
            curr_optim.zero_grad()
            dt.backward()
            nn.utils.clip_grad_norm_(curr_critic.parameters(), 10)
            curr_optim.step()

        # --- Actor Update ---
        # The actor is updated ONLY against its primary objective (self.scalar_weight).
        self.actor_optim.zero_grad()

        # 1. Select the critic corresponding to the actor's primary objective
        if self.args.multi_critics:
            main_idx = np.argmax(self.scalar_weight)
            actor_critic = self.critics[main_idx]
        else:
            actor_critic = self.critic

        # 2. Prepare the conditioned state for the actor's update
        # We must manually construct the state with the *primary* weight attached
        if self.args.weight_conditioned:
            main_weight_tensor = torch.FloatTensor(self.scalar_weight).to(self.args.device)
            main_weight_batch = main_weight_tensor.expand(state_batch.shape[0], -1)
            state_for_actor_critic = torch.cat([state_batch, main_weight_batch], dim=1)
        else:
            state_for_actor_critic = state_batch
            
        # 3. Compute Policy Loss
        # Actor generates actions based on raw state
        actor_actions = self.actor.forward(state_batch)
        
        # Critic evaluates the actor's actions using the conditioned state
        policy_grad_loss = -(actor_critic.forward(state_for_actor_critic, actor_actions)).mean()
        policy_loss = policy_grad_loss

        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optim.step()

        # --- Soft Update Target Networks ---
        soft_update(self.actor_target, self.actor, self.tau)
        
        if self.args.multi_critics:
            for c, c_t in zip(self.critics, self.critics_target):
                soft_update(c_t, c, self.tau)
        else:
            soft_update(self.critic_target, self.critic, self.tau)
        
        return policy_grad_loss.data.cpu().numpy(), delta.data.cpu().numpy()

    def save_info(self, folder_path):
        checkpoint = os.path.join(folder_path, "state_dicts.pkl")
        save_dict = {
            'actor': self.actor.state_dict(),
            'actor_t': self.actor_target.state_dict(),
            'actor_op': self.actor_optim.state_dict(),
        }
        
        if self.args.multi_critics:
            save_dict['critics'] = [c.state_dict() for c in self.critics]
            save_dict['critics_t'] = [c.state_dict() for c in self.critics_target]
            save_dict['critics_op'] = [op.state_dict() for op in self.critics_optims]
        else:
            save_dict['critic'] = self.critic.state_dict()
            save_dict['critic_t'] = self.critic_target.state_dict()
            save_dict['critic_op'] = self.critic_optim.state_dict()

        torch.save(save_dict, checkpoint)
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
        
        if self.args.multi_critics:
            # Assuming args.num_objectives hasn't changed
            for i in range(len(self.critics)):
                self.critics[i].load_state_dict(checkpoint_sd['critics'][i])
                self.critics_target[i].load_state_dict(checkpoint_sd['critics_t'][i])
                self.critics_optims[i].load_state_dict(checkpoint_sd['critics_op'][i])
        else:
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
