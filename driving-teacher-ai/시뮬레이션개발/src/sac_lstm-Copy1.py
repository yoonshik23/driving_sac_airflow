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

from .utils import state_to_tensor
# Hyperparameters
gamma = 0.99
tau = 0.005
alpha = 0.2
lr = 0.0003
buffer_capacity = 1000000
batch_size = 256
hidden_dim = 256
lstm_hidden_dim = 128
embedding_dim = 64
sequence_state_dim = 10

며칠뒤 = 30

# Actor Network
class Actor(nn.Module):
    def __init__(self, device, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.device = device
        self.embedding = nn.Linear(6, 128)
        self.lstm = nn.LSTM(25, 128, batch_first=True)
        self.fc1 = nn.Linear(128 + 128, hidden_dim)  # Adjusted the input size
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.apply(weights_init)

    def forward(self, continuous_vars, variable_length_tensors):
        x1 = torch.relu(self.embedding(continuous_vars))  # (batch_size, 128)

        variable_length_tensors = variable_length_tensors.permute(0, 2, 1)  # (batch_size, max_length, 25)
        x2, _ = self.lstm(variable_length_tensors)  # (batch_size, max_length, 128)
        x2 = x2[:, -1, :]  # 마지막 LSTM 출력 (batch_size, 128)

        x = torch.cat([x1, x2], dim=1)  # (batch_size, 128 + 128)
        x = torch.relu(self.fc1(x))
        mu = self.fc2(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

# Critic Network
class Critic(nn.Module):
    def __init__(self, device, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.device = device
        self.embedding = nn.Linear(6, 128)
        self.lstm = nn.LSTM(25, 128, batch_first=True)
        self.fc1 = nn.Linear(128 + 128 + action_dim, hidden_dim)  # Adjusted the input size
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init)


    def forward(self, continuous_vars, variable_length_tensors, action):
        x1 = torch.relu(self.embedding(continuous_vars))  # (batch_size, 128)

        variable_length_tensors = variable_length_tensors.permute(0, 2, 1)  # (batch_size, max_length, 25)
        x2, _ = self.lstm(variable_length_tensors)  # (batch_size, max_length, 128)
        x2 = x2[:, -1, :]  # 마지막 LSTM 출력 (batch_size, 128)
        # print(action.size())
        x = torch.cat([x1, x2, action], dim=1)  # (batch_size, 128 + 128 + action_dim)
        x = torch.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value

# SAC Agent
class SACAgent:
    def __init__(self, device, action_dim, hidden_dim=256, lr=3e-4, num_actors = 30, num_cars = 11):
        self.device = device 
        self.actor = Actor(device, action_dim, hidden_dim).to(device)
        # self.actors = []
        # for _ in range(며칠뒤):
        #     self.actors.append(Actor(device, action_dim, hidden_dim).to(device))
        self.critic1 = Critic(device, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(device, action_dim, hidden_dim).to(device)
        self.target_critic1 = Critic(device, action_dim, hidden_dim).to(device)
        self.target_critic2 = Critic(device, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        # self.actor_optimizers = []
        # for _ in range(며칠뒤):
        #     self.actor_optimizers.append(optim.Adam(self.actors[i].parameters(), lr=lr))
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.replay_buffer = deque(maxlen=buffer_capacity)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.num_cars = num_cars
        self.q_values = []
        self.critic1_losses = []
        self.critic2_losses = []
        self.actor_losses = []
    
    def select_action(self, state):
        actions = []
        
        for i in range(며칠뒤):
            self.actor.eval()
            # self.actors[i].eval()
            with torch.no_grad():
                continuous_vars, variable_length_tensors = state_to_tensor(state[i], self.device, self.num_cars)
                mu, std = self.actor(continuous_vars, variable_length_tensors)
                # mu, std = self.actors[i](continuous_vars, variable_length_tensors)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
                action = torch.tanh(action)
            self.actor.train()
            # self.actors[i].train()
            actions.append(action.cpu().numpy())
        return actions

    def update(self, batch_size): #####################
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        for i in range(며칠뒤):

            states2 = [state_to_tensor(state, self.device, self.num_cars) for state in states[i]]
            next_states2 = [state_to_tensor(state, self.device, self.num_cars) for state in next_states[i]]
    
            continuous_vars = torch.cat([s[0] for s in states2], dim=0).to(self.device)
            variable_length_tensors = torch.cat([s[1] for s in states2], dim=0).to(self.device)
    
            next_continuous_vars = torch.cat([s[0] for s in next_states2], dim=0).to(self.device)
            next_variable_length_tensors = torch.cat([s[1] for s in next_states2], dim=0).to(self.device)
    
    
            actions2 = torch.tensor(actions[i], dtype=torch.float32).to(self.device).squeeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
    
            with torch.no_grad():
                next_mu, next_std = self.actor(next_continuous_vars, next_variable_length_tensors)
                next_dist = torch.distributions.Normal(next_mu, next_std)
                next_actions = next_dist.rsample()
                next_log_probs = next_dist.log_prob(next_actions).sum(-1, keepdim=True)
                next_q1 = self.target_critic1(next_continuous_vars, next_variable_length_tensors, next_actions)
                next_q2 = self.target_critic2(next_continuous_vars, next_variable_length_tensors, next_actions)
                next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
                target_q = rewards + (1 - dones) * self.gamma * next_q
    
            q1 = self.critic1(continuous_vars, variable_length_tensors, actions2)
            q2 = self.critic2(continuous_vars, variable_length_tensors, actions2)
            critic1_loss = nn.MSELoss()(q1, target_q)
            critic2_loss = nn.MSELoss()(q2, target_q)
    
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
    
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()
    
            mu, std = self.actor(continuous_vars, variable_length_tensors)
            dist = torch.distributions.Normal(mu, std)
            new_actions = dist.rsample()
            log_probs = dist.log_prob(new_actions).sum(-1, keepdim=True)
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
        self.replay_buffer.append(transition)


    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict()
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

    def log_q_values(self, states):
        q_values = []
        for state in states:
            continuous_vars, variable_length_tensors = state_to_tensor(state, self.device)
            with torch.no_grad():
                q1 = self.critic1(continuous_vars, variable_length_tensors, self.actor(continuous_vars, variable_length_tensors)[0])
                q2 = self.critic2(continuous_vars, variable_length_tensors, self.actor(continuous_vars, variable_length_tensors)[0])
                avg_q = (q1 + q2) / 2.0
            q_values.append(avg_q.item())
        self.q_values.append(q_values)

    def plot_metrics(self):
        # Calculate mean and std for q_values
        q_values_mean = [np.mean(q) for q in self.q_values]
        q_values_std = [np.std(q) for q in self.q_values]
        
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
        plt.plot(self.critic1_losses, label='Critic 1 Loss')
        plt.plot(self.critic2_losses, label='Critic 2 Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.title('Critic Loss vs. Update Step')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(self.actor_losses, label='Actor Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.title('Actor Loss vs. Update Step')
        plt.legend()

        plt.tight_layout()
        plt.show()



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

def state_to_tensor(state, device, max_length = 12):
    '''
    [연속형변수, 가변변수]
    '''
    continuous_vars_tensor = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(device)
    variable_length_tensors = [torch.tensor(lst, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device) for lst in state[1]]
    
    # 모든 tensor의 길이가 0인 경우 처리
    if all(tensor.size(1) == 0 for tensor in variable_length_tensors):
        variable_length_tensors = [torch.zeros(1, 1, 1, dtype=torch.float32).to(device) for _ in variable_length_tensors]
    
    variable_length_tensors = pad_variable_length_tensors(variable_length_tensors, device, max_length)
    return continuous_vars_tensor, variable_length_tensors

# 패딩 함수
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