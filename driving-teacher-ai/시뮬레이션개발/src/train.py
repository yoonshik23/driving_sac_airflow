import torch

# Hyperparameters
action_dim = 25
hidden_dim = 256
lr = 3e-4
batch_size = 64
num_episodes = 1000

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize environment and agent
env = CustomEnv()
agent = SACAgent(action_dim, hidden_dim, lr)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.store_transition((state, action, reward, next_state, done))
        agent.update(batch_size)
        
        state = next_state
        episode_reward += reward
    
    print(f"Episode {episode + 1}, Reward: {episode_reward}")