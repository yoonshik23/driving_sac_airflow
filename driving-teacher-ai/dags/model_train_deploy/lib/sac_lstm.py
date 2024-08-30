import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import deque
import matplotlib.pyplot as plt
import torch.nn.init as init
import math
import os
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Hyperparameters
gamma = 0.99
tau = 0.005
alpha = 0.2
lr = 0.0001
buffer_capacity1 = 1000000
buffer_capacity2 = 1500
batch_size = 256
hidden_dim = 256
lstm_hidden_dim = 128
embedding_dim = 64
sequence_state_dim = 10

priority_alpha = 0.6
priority_beta_start = 0.4
priority_beta_frames = 11000
epsilon = 1e-6

며칠뒤 = 30

# SumTree data structure
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (states,actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-6

    def add(self, error, sample):
        p = (error + self.epsilon) ** self.alpha
        self.tree.add(p, sample)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.total() * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (states, actions, rewards, next_states, dones, idxs, is_weight)

    def update(self, idx, error):
        p = (error + self.epsilon) ** self.alpha
        self.tree.update(idx, p)
    def __len__(self):
        return self.tree.agg_write
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.agg_write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        self.agg_write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

# def normalize_state(continuous_vars, variable_length_tensors, update=True):
#     if update:
#         continuous_rms.update(continuous_vars)
#         normalized_continuous_vars = continuous_rms.normalize(continuous_vars)
        
#         normalized_variable_length_tensors = []
#         for tensor in variable_length_tensors:
#             variable_length_rms.update(tensor)
#             normalized_tensor = variable_length_rms.normalize(tensor)
#             normalized_variable_length_tensors.append(normalized_tensor)
#     else:
#         normalized_continuous_vars = continuous_rms.normalize(continuous_vars)
#         normalized_variable_length_tensors = [variable_length_rms.normalize(tensor) for tensor in variable_length_tensors]
    
#     return normalized_continuous_vars, normalized_variable_length_tensors

def pad_variable_length_tensors(tensors, device, max_length=12):
    padded_tensors = []
    for tensor in tensors:
        padding_size = max_length - tensor.size(1)
        if padding_size > 0:
            padding = torch.zeros((tensor.size(0), padding_size), dtype=torch.float32).to(device)
            padded_tensor = torch.cat([tensor.squeeze(-1), padding], dim=1)
        else:
            padded_tensor = tensor.squeeze(-1)
        padded_tensors.append(padded_tensor)
    padded_tensors = torch.stack(padded_tensors, dim=1)  # (batch_size, 25, max_length)
    return padded_tensors

def state_to_tensor(state, device, max_length=12, update=True):
    continuous_vars_tensor = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(device)
    variable_length_tensors = [torch.tensor(lst, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device) for lst in state[1]]
    
    if all(tensor.size(1) == 0 for tensor in variable_length_tensors):
        variable_length_tensors = [torch.zeros(1, 1, 1, dtype=torch.float32).to(device) for _ in variable_length_tensors]
    
    padded_variable_length_tensors = pad_variable_length_tensors(variable_length_tensors, device, max_length)

    # continuous_vars_tensor = torch.tensor(normalized_continuous_vars, dtype=torch.float32).to(device)
    variable_length_tensors = padded_variable_length_tensors.clone().detach().to(device)
    
    return continuous_vars_tensor, variable_length_tensors

def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                init.zeros_(param.data)
class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out
# Actor Network
class Actor(nn.Module):
    def __init__(self, device, action_dim, hidden_dim=256, num_residual_blocks=3, num_cars = 12):
        super(Actor, self).__init__()
        self.device = device
        self.embedding_continuous = nn.Linear(31, hidden_dim)

        self.flatten_variable = nn.Linear(25 * num_cars, hidden_dim)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)])
        self.fc_concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.apply(weights_init)

    def forward(self, continuous_vars, variable_length_tensors):
        x1 = torch.relu(self.embedding_continuous(continuous_vars))
        x2 = variable_length_tensors.view(variable_length_tensors.size(0), -1)  # 평탄화
        x2 = torch.relu(self.flatten_variable(x2))

        x = torch.cat([x1, x2], dim=1)
        x = torch.relu(self.fc_concat(x))
        x = self.residual_blocks(x)
        x = torch.relu(self.fc1(x))
        mu = self.fc2(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std
        
# Critic Network
class Critic(nn.Module):
    def __init__(self, device, action_dim, hidden_dim=256, num_residual_blocks=3,num_cars = 12):
        super(Critic, self).__init__()
        self.device = device
        self.embedding_continuous = nn.Linear(31, hidden_dim)
        # self.embedding_variable = nn.Linear(25, hidden_dim)
        self.fc_concat = nn.Linear(hidden_dim * 2 + action_dim, hidden_dim)
        
        self.flatten_variable = nn.Linear(25 * num_cars, hidden_dim)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init)

    def forward(self, continuous_vars, variable_length_tensors, action):
        x1 = torch.relu(self.embedding_continuous(continuous_vars))
        x2 = variable_length_tensors.view(variable_length_tensors.size(0), -1)  # 평탄화
        x2 = torch.relu(self.flatten_variable(x2))

        x = torch.cat([x1, x2, action], dim=1)
        x = torch.relu(self.fc_concat(x))
        x = self.residual_blocks(x)
        x = torch.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value
class TanhTransform(torch.distributions.transforms.Transform):
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(torch.distributions.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = torch.distributions.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

# SAC Agent
class SACAgent:
    def __init__(self, device, action_dim, hidden_dim=256, lr=3e-4, num_actors=30, num_cars=11):
        self.device = device

        self.actor = Actor(device, action_dim, hidden_dim, num_cars = num_cars).to(device)
        self.critic1 = Critic(device, action_dim, hidden_dim, num_cars = num_cars).to(device)
        self.critic2 = Critic(device, action_dim, hidden_dim, num_cars = num_cars).to(device)
        self.target_critic1 = Critic(device, action_dim, hidden_dim,num_cars = num_cars).to(device)
        self.target_critic2 = Critic(device, action_dim, hidden_dim, num_cars = num_cars).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.replay_buffer1 = PrioritizedReplayBuffer(buffer_capacity1, priority_alpha)
        self.replay_buffer2 = ReplayBuffer(buffer_capacity2)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.num_cars = num_cars
        self.q_values = []
        self.critic1_losses = []
        self.critic2_losses = []
        self.actor_losses = []

        # self.continuous_rms = RunningMeanStd(31)
        # self.variable_length_rms = RunningMeanStd(self.num_cars)
    
    def select_action(self, state):
        actions = []
        for i in range(len(state)):
            self.actor.eval()
            with torch.no_grad():
                continuous_vars, variable_length_tensors = state_to_tensor(
                    state[i], self.device, self.num_cars, update=False)
                mu, std = self.actor(continuous_vars, variable_length_tensors)
                dist = SquashedNormal(mu, std)
                action = dist.sample()
            self.actor.train()
            actions.append(action.cpu().numpy())
        return actions


    def update(self, batch_size, frame_idx):
        if len(self.replay_buffer1) < batch_size:
            return
        # # states, actions, rewards, next_states, dones = self.replay_buffer1.sample(batch_size)
        # states1, actions1, rewards1, next_states1, dones1 = self.replay_buffer1.sample(int(batch_size/2))
        # states2, actions2, rewards2, next_states2, dones2 = self.replay_buffer2.sample(int(batch_size/2))
        # states = states1 + states2
        # actions = actions1 + actions2
        # rewards = rewards1 + rewards2
        # next_states = next_states1 + next_states2
        # dones = dones1 + dones2
        
        beta = min(1.0, priority_beta_start + frame_idx * (1.0 - priority_beta_start) / priority_beta_frames)
        
        # Prioritized sampling from replay buffer
        states, actions, rewards, next_states, dones, idxs, is_weights = self.replay_buffer1.sample(batch_size, beta)
        is_weights = torch.tensor(is_weights, dtype=torch.float32).to(self.device)
        


        states2 = [state_to_tensor(state, self.device, self.num_cars) for state in states]
        next_states2 = [state_to_tensor(state, self.device, self.num_cars) for state in next_states]

        continuous_vars = torch.cat([s[0] for s in states2], dim=0).to(self.device)
        variable_length_tensors = torch.cat([s[1] for s in states2], dim=0).to(self.device)

        next_continuous_vars = torch.cat([s[0] for s in next_states2], dim=0).to(self.device)
        next_variable_length_tensors = torch.cat([s[1] for s in next_states2], dim=0).to(self.device)

        actions2 = torch.tensor(actions, dtype=torch.float32).to(self.device).squeeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones2 = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_mu, next_std = self.actor(next_continuous_vars, next_variable_length_tensors)
            next_dist = SquashedNormal(next_mu, next_std)
            next_actions = next_dist.rsample()
            next_log_probs = next_dist.log_prob(next_actions).sum(-1, keepdim=True) - torch.log(1 - next_actions.pow(2) + 1e-6).sum(-1, keepdim=True)
            next_q1 = self.target_critic1(next_continuous_vars, next_variable_length_tensors, next_actions)
            next_q2 = self.target_critic2(next_continuous_vars, next_variable_length_tensors, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones2) * self.gamma * next_q

        q1 = self.critic1(continuous_vars, variable_length_tensors, actions2)
        q2 = self.critic2(continuous_vars, variable_length_tensors, actions2)
        critic1_loss = (is_weights * (q1 - target_q).pow(2)).mean()
        critic2_loss = (is_weights * (q2 - target_q).pow(2)).mean()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        mu, std = self.actor(continuous_vars, variable_length_tensors)
        dist = SquashedNormal(mu, std)
        new_actions = dist.rsample()
        log_probs = dist.log_prob(new_actions).sum(-1, keepdim=True) - torch.log(1 - new_actions.pow(2) + 1e-6).sum(-1, keepdim=True)
        q1 = self.critic1(continuous_vars, variable_length_tensors, new_actions)
        q2 = self.critic2(continuous_vars, variable_length_tensors, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic1_losses.append(critic1_loss.item())
        self.critic2_losses.append(critic2_loss.item())
        self.actor_losses.append(actor_loss.item())

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, transition):
        state, action, reward, next_state, done = transition

        with torch.no_grad():
            continuous_vars, variable_length_tensors = state_to_tensor(
                state, self.device, self.num_cars, update=False)
            mu, std = self.actor(continuous_vars, variable_length_tensors)
            dist = SquashedNormal(mu, std)
            action = dist.sample()


        # 현재 상태에서 TD 오류를 계산하기 위해 상태를 텐서로 변환
        continuous_vars, variable_length_tensors = state_to_tensor(state, self.device, self.num_cars, update=False)
        next_continuous_vars, next_variable_length_tensors = state_to_tensor(next_state, self.device, self.num_cars, update=False)
        
        with torch.no_grad():
            action_tensor = action.clone().detach().to(self.device).squeeze(1)
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)
            done_tensor = torch.tensor([done], dtype=torch.float32).to(self.device)
            
            next_mu, next_std = self.actor(next_continuous_vars, next_variable_length_tensors)
            next_dist = SquashedNormal(next_mu, next_std)
            next_action = next_dist.sample()
            next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True) - torch.log(1 - next_action.pow(2) + 1e-6).sum(-1, keepdim=True)
            next_q1 = self.target_critic1(next_continuous_vars, next_variable_length_tensors, next_action)
            next_q2 = self.target_critic2(next_continuous_vars, next_variable_length_tensors, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * next_q

            current_q1 = self.critic1(continuous_vars, variable_length_tensors, action_tensor)
            current_q2 = self.critic2(continuous_vars, variable_length_tensors, action_tensor)

            td_error = torch.abs(current_q1 - target_q).item() + torch.abs(current_q2 - target_q).item()

        self.replay_buffer1.add(td_error, transition)
        self.replay_buffer2.add(transition)

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            # 'continuous_rms_mean': self.continuous_rms.mean,
            # 'continuous_rms_var': self.continuous_rms.var,
            # 'continuous_rms_count': self.continuous_rms.count,
            # 'variable_length_rms_mean': self.variable_length_rms.mean,
            # 'variable_length_rms_var': self.variable_length_rms.var,
            # 'variable_length_rms_count': self.variable_length_rms.count
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        # self.continuous_rms.mean = checkpoint['continuous_rms_mean']
        # self.continuous_rms.var = checkpoint['continuous_rms_var']
        # self.continuous_rms.count = checkpoint['continuous_rms_count']
        # self.variable_length_rms.mean = checkpoint['variable_length_rms_mean']
        # self.variable_length_rms.var = checkpoint['variable_length_rms_var']
        # self.variable_length_rms.count = checkpoint['variable_length_rms_count']

    def log_q_values(self, states):
        q_values = []
        for state in states:
            continuous_vars, variable_length_tensors = state_to_tensor(state, self.device, self.num_cars, update=False)
            with torch.no_grad():
                q1 = self.critic1(continuous_vars, variable_length_tensors, self.actor(continuous_vars, variable_length_tensors)[0])
                q2 = self.critic2(continuous_vars, variable_length_tensors, self.actor(continuous_vars, variable_length_tensors)[0])
                avg_q = (q1 + q2) / 2.0
            q_values.append(avg_q.item())
        self.q_values.append(q_values)
        print('q_values: ', round(np.mean(q_values), 3))

    def plot_metrics(self):
        # Calculate mean and std for q_values
        if len(self.q_values) > 1:
            q_values_mean = [np.mean(q) for q in self.q_values[-int(len(self.q_values)/2):]]
            q_values_std = [np.std(q) for q in self.q_values[-int(len(self.q_values)/2):]]
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(4, 1, 1)
            plt.plot(q_values_mean, label='Mean Q Value')
            plt.fill_between(range(len(q_values_mean)), 
                             np.array(q_values_mean) - np.array(q_values_std), 
                             np.array(q_values_mean) + np.array(q_values_std), 
                             alpha=0.3, label='Q Value Std Dev')
            plt.xlabel('Episode')
            plt.ylabel('Q Value')
            plt.title('Average Q Value vs. Episode')
            plt.legend()
    
            plt.subplot(4, 1, 2)
            plt.plot(self.critic1_losses[-int(len(self.critic1_losses)/2):], label='Critic 1 Loss')
            plt.plot(self.critic2_losses[-int(len(self.critic2_losses)/2):], label='Critic 2 Loss')
            plt.xlabel('Update Step')
            plt.ylabel('Loss')
            plt.title('Critic Loss vs. Update Step')
            plt.legend()
    
            plt.subplot(4, 1, 3)
            plt.plot(self.actor_losses[-int(len(self.actor_losses)/2):], label='Actor Loss')
            plt.xlabel('Update Step')
            plt.ylabel('Loss')
            plt.title('Actor Loss vs. Update Step')
            plt.legend()
    
            plt.tight_layout()
            plt.show()

