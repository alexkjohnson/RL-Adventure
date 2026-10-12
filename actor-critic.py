import math
import random

import gymnasium as gym
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.distributions import Categorical

import matplotlib.pyplot as plt 

# Use Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create environments
num_envs = 16
env_name = "CartPole-v1"

envs = gym.make_vec(env_name, num_envs=num_envs) # Used for parallel training
env = gym.make(env_name)             # Single env: Used for evaluation/testing

# Neural Network
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value

# Helper Functions
def test_env():
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            dist, _ = model(state_t)
        
        action = dist.sample().cpu().numpy()[0]
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    return total_reward

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.n
hidden_size = 256
lr          = 3e-4
num_steps   = 5

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames = 20000
frame_idx = 0
test_rewards = []

# Training Loop
state, _ = envs.reset()

while frame_idx < max_frames:
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0

    for _ in range(num_steps):
        state_t = torch.FloatTensor(state).to(device)
        dist, value = model(state_t)
        action = dist.sample()

        # Gymnasium step returns: next_state, reward, terminated, truncated, info
        next_state, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1.0 - done.astype(float)).unsqueeze(1).to(device))

        state = next_state
        frame_idx += num_envs

    # Evaluate performance periodically
    if (frame_idx // num_envs) % 50 == 0:
        mean_reward = np.mean([test_env() for _ in range(5)])
        test_rewards.append(mean_reward)
        print(f"Frames: {frame_idx} | Average Test Reward: {mean_reward:.1f}")

    next_state_t = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state_t)
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns).detach()
    values    = torch.cat(values)

    advantage = returns - values
    
    actor_loss  = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()
    
    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training finished successfully!")
final_score = test_env()
print(f"Final Trained Agent Reward: {final_score}")