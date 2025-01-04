"""
on the piano_with_shadow_hands.PianoWithShadowHands task, this should learn to play the 
"Eye Water" midi file.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
from robopianist.suite import load
from robopianist import music
from dm_env_wrappers import CanonicalSpecWrapper

# Actor network for PPO
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        mean = self.network(state)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

# Critic network for PPO 
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, state):
        return self.network(state)

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        state = torch.FloatTensor(state)
        dist = self.actor(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.detach().numpy(), log_prob.detach()

    def update(self, states, actions, rewards, log_probs, next_states):
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.FloatTensor(log_probs)
        next_states = torch.FloatTensor(next_states)

        # Compute advantages
        values = self.critic(states).detach()
        next_values = self.critic(next_states).detach()
        advantages = rewards + self.gamma * next_values - values

        # PPO update
        for _ in range(5):  # Multiple epochs
            # Actor update
            dist = self.actor(states)
            new_log_probs = dist.log_prob(actions).sum(1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Critic update
            value_pred = self.critic(states)
            value_target = rewards + self.gamma * next_values
            critic_loss = nn.MSELoss()(value_pred, value_target)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

def main():
    # Load environment
    env = load(
        environment_name="piano_with_shadow_hands",
        midi_file="Sword Art Online - Crossing Field (SAO OP).mid",
        task_kwargs={
            "change_color_on_activation": True,
            "trim_silence": True,
            "control_timestep": 0.05,
            "gravity_compensation": True,
        }
    )
    env = CanonicalSpecWrapper(env)

    # Get dimensions
    obs = env.reset()
    state_dim = sum(np.prod(v.shape) for v in obs.observation.values())
    action_dim = env.action_spec().shape[0]

    # Initialize agent
    agent = PPOAgent(state_dim, action_dim)
    
    # Training loop
    num_episodes = 1000
    for episode in range(num_episodes):
        states, actions, rewards, log_probs, next_states = [], [], [], [], []
        
        timestep = env.reset()
        episode_reward = 0
        
        while not timestep.last():
            state = np.concatenate([v.flatten() for v in timestep.observation.values()])
            action, log_prob = agent.select_action(state)
            
            next_timestep = env.step(action)
            reward = next_timestep.reward
            next_state = np.concatenate([v.flatten() for v in next_timestep.observation.values()])
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            next_states.append(next_state)
            
            episode_reward += reward
            timestep = next_timestep

        # Update agent
        agent.update(states, actions, rewards, log_probs, next_states)
        
        print(f"Episode {episode}, Reward: {episode_reward}")

if __name__ == "__main__":
    main()