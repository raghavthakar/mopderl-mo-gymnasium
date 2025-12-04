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


class MOCritic(nn.Module):
    """
    A unified facade for the Critic architecture. 
    It manages whether the agent uses a single conditioned critic or 
    multiple specialist critics based on the provided arguments.
    """
    def __init__(self, args: Parameters):
        super(MOCritic, self).__init__()
        self.args = args
        self.num_objectives = args.num_objectives

        # Enforce exclusivity: strictly separate architectures
        if self.args.multi_critics and self.args.weight_conditioned:
            raise ValueError(
                "Ambiguous Critic Configuration: 'multi_critics' and 'weight_conditioned' "
                "cannot both be True. Choose one approach."
            )

        if self.args.multi_critics:
            # Multi-Critic Mode: One distinct network per objective.
            # Mapping: Index i in this list corresponds to Objective i.
            self.specialists = nn.ModuleList([Critic(args) for _ in range(self.num_objectives)])
            self.mode = "multi"
        else:
            # Single Critic Mode: One network for all objectives.
            # If weight_conditioned is True, the Critic class handles input expansion internally.
            self.generalist = Critic(args)
            self.mode = "single"

        self.to(args.device)

    def forward(self, state, action, scalar_weight):
        """
        Routes the forward pass to the correct network(s) based on configuration.
        
        Args:
            state: Raw state tensor [batch, state_dim]
            action: Action tensor [batch, action_dim]
            scalar_weight: The target scalarisation weight [batch, num_obj] or [num_obj]
        """
        # Ensure weight is a tensor for conditioning or indexing
        if not isinstance(scalar_weight, torch.Tensor):
            scalar_weight = torch.FloatTensor(scalar_weight).to(self.args.device)
        
        # Expand scalar_weight to batch size if necessary
        if scalar_weight.dim() == 1:
            scalar_weight = scalar_weight.expand(state.shape[0], -1)

        if self.mode == "multi":
            # 1. Identify the target objective index from the one-hot weight vector.
            # Assumption: The weight vector is one-hot (e.g. [0, 1]).
            # We take the argmax to find the active objective index.
            # We use the first element of the batch since the batch is uniform for one update.
            obj_idx = torch.argmax(scalar_weight[0]).item()
            
            # 2. Select the specialist corresponding to that objective.
            active_critic = self.specialists[obj_idx]
            
            # 3. Forward pass (Raw State).
            return active_critic(state, action)

        else:
            # Single Critic Mode (Conditioned or Standard)
            if self.args.weight_conditioned:
                # Concatenate state and weight vector
                state_in = torch.cat([state, scalar_weight], dim=1)
            else:
                # Standard DDPG (Raw State)
                state_in = state
            
            return self.generalist(state_in, action)


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
        """
        Updates the actor based on the provided parents and critic.
        The 'critic' argument is expected to be an instance of MOCritic.
        """
        state_batch, _, _, _, _ = batch

        # --- Get Q-values ---
        # We pass the raw state and scalar_weight to the MOCritic.
        # The MOCritic handles any necessary concatenation or network selection internally.
        
        # Parents act on raw state
        p1_action = p1(state_batch)
        p2_action = p2(state_batch)
        
        # Critic evaluates using the passed scalar_weight context
        p1_q = critic(state_batch, p1_action, scalar_weight).flatten()
        p2_q = critic(state_batch, p2_action, scalar_weight).flatten()

        # --- Filter actions and states ---
        eps = 0.0
        # Select actions where the respective parent had a higher Q-value
        mask_p1 = (p1_q - p2_q > eps)
        mask_p2 = (p2_q - p1_q >= eps)
        
        action_batch = torch.cat((p1_action[mask_p1], p2_action[mask_p2])).detach()
        state_batch_filtered = torch.cat((state_batch[mask_p1], state_batch[mask_p2]))
        
        # --- Actor Update ---
        # The agent's actor predicts actions on the filtered states
        actor_action = self.actor(state_batch_filtered)

        self.actor_optim.zero_grad()
        # Loss: Match the better parent's action + regularization
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

        # Construct input interface
        # Logic: If weight_conditioned is True, we augment state dim.
        # If multi_critics is True (and therefore weight_conditioned is False), we use raw state dim.
        input_dim = args.state_dim + args.num_objectives if args.weight_conditioned else args.state_dim
        
        self.w_state_l1 = nn.Linear(input_dim, l1)
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

        # --- MOCritic Initialization ---
        self.critic = MOCritic(args)
        self.critic_target = MOCritic(args)
        
        # --- Optimizer Initialization ---
        # While MOCritic abstracts the forward pass, DDPG must manage optimizers for training.
        if self.args.multi_critics:
            # Create a separate optimizer for each specialist network in the MOCritic
            self.critics_optims = [Adam(c.parameters(), lr=0.5e-3) for c in self.critic.specialists]
        else:
            # Single optimizer for the generalist network
            self.critic_optim = Adam(self.critic.parameters(), lr=0.5e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic) # MOCritic recursively handles state_dict updates

        if checkpoint_folder is not None:
            self.load_info(checkpoint_folder)
            self.actor.train()
            self.actor_target.train()
            self.critic.train()
            self.critic_target.train()

    def update_parameters(self, batch):
        state_batch, action_batch, reward_vec, next_state_batch, done_batch = batch
        
        state_batch = state_batch.to(self.args.device)
        action_batch = action_batch.to(self.args.device)
        reward_vec = reward_vec.to(self.args.device) 
        next_state_batch = next_state_batch.to(self.args.device)
        
        if self.args.use_done_mask:
            done_batch = done_batch.to(self.args.device)
        
        delta = torch.tensor(0.0)
        
        self.actor_target.to(self.args.device)
        self.critic.to(self.args.device)
        self.critic_target.to(self.args.device)

        # List of weights to train on
        weight_vectors = np.concatenate(([self.scalar_weight], self.other_weights), axis=0) if (self.args.weight_conditioned or self.args.multi_critics) else [self.scalar_weight]

        # --- Critic Update Loop ---
        for weight_np in weight_vectors:
            
            weight_tensor = torch.FloatTensor(weight_np).to(self.args.device)
            weight_batch = weight_tensor.expand(state_batch.shape[0], -1)

            # Scalarize reward
            scalar_reward_batch = torch.matmul(reward_vec, weight_tensor.reshape(-1, 1))

            # Select the correct optimizer for this specific weight context
            if self.args.multi_critics:
                obj_idx = np.argmax(weight_np)
                curr_optim = self.critics_optims[obj_idx]
                curr_network_params = self.critic.specialists[obj_idx].parameters()
            else:
                curr_optim = self.critic_optim
                curr_network_params = self.critic.parameters()

            # --- Compute Target ---
            next_action_batch = self.actor_target.forward(next_state_batch)
            
            with torch.no_grad():
                # MOCritic handles routing/conditioning automatically
                next_q = self.critic_target(next_state_batch, next_action_batch, weight_batch)
                
                if self.args.use_done_mask: 
                    next_q = next_q * (1 - done_batch)
                target_q = scalar_reward_batch + (self.gamma * next_q)

            # --- Compute Loss & Update ---
            # MOCritic handles routing/conditioning automatically
            current_q = self.critic(state_batch, action_batch, weight_batch)
            
            delta = (current_q - target_q).abs()
            dt = torch.mean(delta**2)
            
            curr_optim.zero_grad()
            dt.backward()
            nn.utils.clip_grad_norm_(curr_network_params, 10)
            curr_optim.step()

        # --- Actor Update ---
        self.actor_optim.zero_grad()
        
        # Prepare weight tensor for the actor's primary objective
        main_weight_tensor = torch.FloatTensor(self.scalar_weight).to(self.args.device)
        main_weight_batch = main_weight_tensor.expand(state_batch.shape[0], -1)

        actor_actions = self.actor.forward(state_batch)
        
        # MOCritic evaluates: V(s, a, primary_weight)
        policy_grad_loss = -(self.critic(state_batch, actor_actions, main_weight_batch)).mean()
        
        policy_grad_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optim.step()

        # --- Soft Update ---
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        
        return policy_grad_loss.data.cpu().numpy(), delta.data.cpu().numpy()

    def save_info(self, folder_path):
        checkpoint = os.path.join(folder_path, "state_dicts.pkl")
        save_dict = {
            'actor': self.actor.state_dict(),
            'actor_t': self.actor_target.state_dict(),
            'actor_op': self.actor_optim.state_dict(),
            # Since MOCritic is an nn.Module, state_dict() automatically saves all sub-modules (lists or single)
            'critic': self.critic.state_dict(),
            'critic_t': self.critic_target.state_dict(),
        }
        
        # Save optimizers manually as they aren't part of the module hierarchy
        if self.args.multi_critics:
            save_dict['critics_op'] = [op.state_dict() for op in self.critics_optims]
        else:
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
        
        # Load MOCritic state
        self.critic.load_state_dict(checkpoint_sd['critic'])
        self.critic_target.load_state_dict(checkpoint_sd['critic_t'])
        
        # Load Optimizers
        if self.args.multi_critics:
            for i, op_sd in enumerate(checkpoint_sd['critics_op']):
                self.critics_optims[i].load_state_dict(op_sd)
        else:
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