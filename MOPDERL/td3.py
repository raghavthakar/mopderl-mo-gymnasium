from .ddpg import *
from .replay_memory import ReplayMemory

class TD3:
    def __init__(self, args, scalar_weight: np.ndarray, checkpoint_folder=None) -> None:
        self.args = args
        self.scalar_weight = scalar_weight
        
        # self.action_lower_bound = -1e6
        # self.action_upper_bound = 1e6
        # if args.env_name == "Hopper-v2":
        #     self.action_lower_bound = torch.tensor([-2.0, -2.0, -4.0], dtype=torch.float32, device=args.device)
        #     self.action_upper_bound = torch.tensor([2.0, 2.0, 4.0], dtype=torch.float32, device=args.device)

        self.buffer = ReplayMemory(args.buffer_size, args.device)

        self.actor = Actor(args, init=True)
        self.actor.ounoise = CustomNoise(scale=0.1, action_dim=args.action_dim, batch_size=args.batch_size)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr=0.5e-4)

        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.5e-3)

        self.critic_p = Critic(args)
        self.critic_p_target = Critic(args)
        self.critic_p_optim = Adam(self.critic_p.parameters(), lr=0.5e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        self.learn_step = 0
        self.actor_update_interval = self.args.actor_update_interval

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        hard_update(self.critic_p_target, self.critic_p)


        if checkpoint_folder is not None:
            self.load_info(checkpoint_folder)
            self.actor.train()
            self.actor_target.train()
            self.critic.train()
            self.critic_target.train()
    
    def update_parameters(self, batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        # Scalarize rewards
        reward_batch = torch.matmul(reward_batch, torch.FloatTensor(np.reshape(self.scalar_weight, (-1, 1))))

        state_batch = state_batch.to(self.args.device)
        next_state_batch = next_state_batch.to(self.args.device)
        action_batch = action_batch.to(self.args.device)
        reward_batch = reward_batch.to(self.args.device)

        # Update critics
        next_action_batch = self.actor_target(next_state_batch) + torch.from_numpy(self.actor.ounoise.noise(training=True)).float()
        # if self.args.env_name == "Hopper-v2":
        #     next_action_batch = torch.max(torch.min(next_action_batch, self.action_upper_bound), self.action_lower_bound)

        next_q = self.critic_target(next_state_batch, next_action_batch)
        next_q_p = self.critic_p_target(next_state_batch, next_action_batch)
        if self.args.use_done_mask:
            next_q = next_q * (1 - done_batch)
            next_q_p = next_q_p * (1 - done_batch)
        target_q = reward_batch + self.gamma * torch.min(next_q, next_q_p)

        current_value = self.critic(state_batch, action_batch)
        current_value_p = self.critic_p(state_batch, action_batch)
        self.critic_optim.zero_grad()
        self.critic_p_optim.zero_grad()
        loss = self.loss(target_q, current_value)
        loss_p = self.loss(target_q, current_value_p)
        loss = loss + loss_p
        loss.backward()
        self.critic_optim.step()
        self.critic_p_optim.step()
        self.learn_step = (self.learn_step + 1) % self.actor_update_interval
        
        if self.learn_step == 0:
            self.actor_optim.zero_grad()
            actor_loss = self.critic(state_batch, self.actor(state_batch))
            actor_loss = -actor_loss.mean()
            actor_loss.backward()
            self.actor_optim.step()
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.critic_p_target, self.critic_p, self.tau)
            return loss.data.cpu().numpy(), actor_loss.data.cpu().numpy()

        return loss.data.cpu().numpy(), -1
    
    def save_info(self, folder_path):
        checkpoint = os.path.join(folder_path, "state_dicts.pkl")
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_t': self.actor_target.state_dict(),
            'actor_op': self.actor_optim.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_t': self.critic_target.state_dict(),
            'critic_op': self.critic_optim.state_dict(),
            'critic_p': self.critic_p.state_dict(),
            'critic_p_t': self.critic_p_target.state_dict(),
            'critic_p_op': self.critic_p_optim.state_dict(),
        }, checkpoint)
        buffer_path = os.path.join(folder_path, "buffer.npy")
        self.buffer.save_info(buffer_path)
        # ou_path  = os.path.join(folder_path, "ou.npy")
        # with open(ou_path, 'wb') as f:
        #     np.save(f, self.actor.ounoise.state)
    
    def load_info(self, folder_path):
        checkpoint = os.path.join(folder_path, "state_dicts.pkl")
        checkpoint_sd = torch.load(checkpoint)
        self.actor.load_state_dict(checkpoint_sd['actor'])
        self.actor_target.load_state_dict(checkpoint_sd['actor_t'])
        self.actor_optim.load_state_dict(checkpoint_sd['actor_op'])
        self.critic.load_state_dict(checkpoint_sd['critic'])
        self.critic_target.load_state_dict(checkpoint_sd['critic_t'])
        self.critic_optim.load_state_dict(checkpoint_sd['critic_op'])
        self.critic_p.load_state_dict(checkpoint_sd['critic_p'])
        self.critic_p_target.load_state_dict(checkpoint_sd['critic_p_t'])
        self.critic_p_optim.load_state_dict(checkpoint_sd['critic_p_op'])
        buffer_path = os.path.join(folder_path, "buffer.npy")
        self.buffer.load_info(buffer_path)
        # ou_path  = os.path.join(folder_path, "ou.npy")
        # with open(ou_path, 'rb') as f:
        #     ou_state = np.load(f)
        #     self.actor.ounoise.state = ou_state


class CustomNoise():
    def __init__(self, scale, action_dim, batch_size) -> None:
        self.scale = scale
        self.action_dim = action_dim
        self.batch_size = batch_size
    
    def noise(self, training=None):
        if training is not None:
            return np.random.normal(scale=self.scale, size=(self.batch_size, self.action_dim))
        return np.random.normal(scale=self.scale, size=(self.action_dim, ))