import math
import random
import argparse

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train A2C on Pendulum-v1 with video recording")
parser.add_argument("--vis", action="store_true", help="Record an evaluation video of the trained agent")
cli_args = parser.parse_args()

# Use cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create environments
num_envs = 16
env_name = "Pendulum-v1"

envs = gym.make_vec(env_name, num_envs=num_envs)
env = gym.make(env_name)

# Neural Network
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(), # Tanh is often preferred over ReLU for continuous control
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, num_outputs), std=0.01) # Small std for initial actions near 0
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs)) # Modern parameterization for log standard deviation

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor_mean(x)
        std   = self.actor_logstd.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

def test_env(vis=False):
    eval_env = gym.make(env_name, render_mode="rgb_array") if vis else env
    if vis:
        eval_env = RecordVideo(eval_env, video_folder="videos", episode_trigger=lambda e: True)

    state, _ = eval_env.reset()
    done = False
    total_reward = 0
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            dist, _ = model(state_t)
            action = dist.sample().cpu().numpy()[0]
            action = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)

        next_state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward

    if vis:
        eval_env.close()
        print("Saved evaluation video to 'videos/' folder!")
    return total_reward

# High-Dimensional Continuous Control Using GAE (Global Scope)
def compute_gae(next_value, rewards, masks, values, gamma=0.99, lam=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]
hidden_size = 256
lr          = 3e-4
num_steps   = 20

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters())

max_frames   = 100000
frame_idx    = 0
test_rewards = []

# Training Loop
state, _ = envs.reset()

while frame_idx < max_frames:
    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    entropy   = 0

    for _ in range(num_steps):
        state_t = torch.FloatTensor(state).to(device)
        dist, value = model(state_t)
        action = dist.sample()

        next_state, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        entropy += dist.entropy().sum(dim=1).mean()
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1.0 - done.astype(float)).unsqueeze(1).to(device))
        
        state = next_state
        frame_idx += num_envs
        
        if (frame_idx // num_envs) % 50 == 0:
            mean_reward = np.mean([test_env() for _ in range(5)])
            test_rewards.append(mean_reward)
            print(f"Frames: {frame_idx} | Average Test Reward: {mean_reward:.1f}")
            
    next_state_t = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state_t)
    returns = compute_gae(next_value, rewards, masks, values)
    
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
final_score = test_env(vis=cli_args.vis)
print(f"Final Trained Agent Reward: {final_score}")