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
parser = argparse.ArgumentParser(description="Train PPO on Pendulum-v1 with video recording")
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
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super().__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs) * std)

        self.apply(init_weights)

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

# Generalized Advantage Estimation (GAE)
def compute_gae(next_value, rewards, masks, values, gamma=0.99, lam=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

# PPO Batch Generator
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()
            new_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]

# Hyperparameters
hidden_size      = 256
lr               = 3e-4
num_steps        = 20
mini_batch_size  = 5
ppo_epochs       = 4
threshold_reward = -200

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames   = 150000  # Increased to give the agent enough time to learn
frame_idx    = 0
test_rewards = []

# Training Loop
state, _ = envs.reset()
early_stop = False

while frame_idx < max_frames and not early_stop:
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []

    for _ in range(num_steps):
        state_t = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            dist, value = model(state_t)
            action = dist.sample()

        next_state, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1.0 - done.astype(float)).unsqueeze(1).to(device))
        
        states.append(state_t)
        actions.append(action)

        state = next_state
        frame_idx += num_envs
        
    if (frame_idx // num_envs) % 50 == 0:
        mean_reward = np.mean([test_env() for _ in range(5)])
        test_rewards.append(mean_reward)
        print(f"Frames: {frame_idx} | Average Test Reward: {mean_reward:.1f}")
        if mean_reward > threshold_reward:
            early_stop = True
            
    with torch.no_grad():
        next_state_t = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state_t)
        
    returns = compute_gae(next_value, rewards, masks, values)
    
    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    
    advantage = returns - values

    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

print("Training finished successfully!")
final_score = test_env(vis=cli_args.vis)
print(f"Final Trained Agent Reward: {final_score}")