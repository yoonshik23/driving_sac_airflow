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
    variable_length_tensors = torch.tensor(padded_variable_length_tensors, dtype=torch.float32).to(device)
    
    return continuous_vars_tensor, variable_length_tensors
    
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
    variable_length_tensors = torch.tensor(padded_variable_length_tensors, dtype=torch.float32).to(device)
    
    return continuous_vars_tensor, variable_length_tensors

class PolicyNetwork(nn.Module):
    def __init__(self, device, action_dim, hidden_dim=256, num_residual_blocks=3, num_cars=12):
        super(PolicyNetwork, self).__init__()
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
        # self.apply(weights_init)

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


class ValueNetwork(nn.Module):
    def __init__(self, device, hidden_dim=256, num_residual_blocks=3, num_cars=12):
        super(ValueNetwork, self).__init__()
        self.device = device
        self.embedding_continuous = nn.Linear(31, hidden_dim)
        self.flatten_variable = nn.Linear(25 * num_cars, hidden_dim)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)])
        self.fc_concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        # self.apply(weights_init)

    def forward(self, continuous_vars, variable_length_tensors):
        x1 = torch.relu(self.embedding_continuous(continuous_vars))
        x2 = variable_length_tensors.view(variable_length_tensors.size(0), -1)  # 평탄화
        x2 = torch.relu(self.flatten_variable(x2))

        x = torch.cat([x1, x2], dim=1)
        x = torch.relu(self.fc_concat(x))
        x = self.residual_blocks(x)
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value

class PPOAgent:
    def __init__(self, device, action_dim, hidden_dim=256, lr=3e-4, num_cars=12):
        self.device = device
        self.policy_net = PolicyNetwork(device, action_dim, hidden_dim, num_cars=num_cars).to(device)
        self.value_net = ValueNetwork(device, hidden_dim, num_cars=num_cars).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.clip_param = 0.2
        self.update_epochs = 10
        self.memory = [[]]*30
        self.value_losses = []
        self.policy_losses = []
        self.num_cars = num_cars

    def select_action(self, state):
        actions = []
        log_probs = []
        for i in range(30):
            self.policy_net.eval()
            with torch.no_grad():
                continuous_vars, variable_length_tensors = state_to_tensor(
                    state[i], self.device, self.num_cars, update=False)
                mu, std = self.policy_net(continuous_vars, variable_length_tensors)
                dist = Normal(mu, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            self.policy_net.train()
            actions.append(action.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
        return actions, log_probs

    def store_transition(self, transition):
        for i in range(30):
            self.memory[i].append((transition[0][i], transition[1][i], transition[2][i], transition[3][i], transition[4][i], transition[5][i]))

    def clear_memory(self):
        self.memory = [[]]*30

    def update(self):
        # memory에서 데이터 추출
        states, actions, rewards, next_states, dones, log_probs = zip(*self.memory.pop(0))
        self.memory.append([])
        states = [state_to_tensor(state, self.device, self.num_cars) for state in states]
        next_states = [state_to_tensor(state, self.device, self.num_cars) for state in next_states]

        continuous_vars = torch.cat([s[0] for s in states], dim=0).to(self.device)
        variable_length_tensors = torch.cat([s[1] for s in states], dim=0).to(self.device)
        next_continuous_vars = torch.cat([s[0] for s in next_states], dim=0).to(self.device)
        next_variable_length_tensors = torch.cat([s[1] for s in next_states], dim=0).to(self.device)

        actions = torch.tensor(actions, dtype=torch.float32).to(self.device).squeeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device).squeeze(1)

        # Value Network 업데이트
        with torch.no_grad():
            next_values = self.value_net(next_continuous_vars, next_variable_length_tensors).squeeze(1)
            target_values = rewards + (1 - dones) * next_values
        values = self.value_net(continuous_vars, variable_length_tensors).squeeze(1)
        value_loss = F.mse_loss(values, target_values)
        self.value_losses.append(value_loss.item())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Policy Network 업데이트
        # for _ in range(self.update_epochs):
        mu, std = self.policy_net(continuous_vars, variable_length_tensors)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(-1)
        advantages = target_values - values
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        print(surr2)
        policy_loss = -torch.min(surr1, surr2).mean()
        self.policy_losses.append(policy_loss.item())
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        print('성공')

        
        

    def save_model(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),

            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),


        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])


    def plot_metrics(self):
        # Calculate mean and std for q_values
        if len(self.value_losses) > 1:
    
            plt.subplot(4, 1, 2)
            plt.plot(self.value_losses[-int(len(self.value_losses)/2):], label='Value  Loss')

            plt.xlabel('Update Step')
            plt.ylabel('Loss')
            plt.title('Value Loss vs. Update Step')
            plt.legend()
    
            plt.subplot(4, 1, 3)
            plt.plot(self.policy_losses[-int(len(self.policy_losses)/2):], label='Policy Loss')
            plt.xlabel('Update Step')
            plt.ylabel('Loss')
            plt.title('Policy Loss vs. Update Step')
            plt.legend()
    
            plt.tight_layout()
            plt.show()


