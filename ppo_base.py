"""
PPO base class
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
from pathlib import Path

# Actor network for PPO
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        
    def forward(self, state):
        mean = self.network(state)
        std = torch.exp(torch.clamp(self.log_std, -20, 2))
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
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, device='cuda', checkpoint_dir='checkpoints'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.max_grad_norm = 0.5
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def select_actions(self, states):
        """Handle batch of states"""
        states = torch.FloatTensor(states).to(self.device)
        dist = self.actor(states)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(1)  # Sum over action dimensions for each batch
        return actions.cpu().detach().numpy(), log_probs.cpu().detach().numpy()

    def update(self, states, actions, rewards, log_probs, next_states):
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # Normalize states and rewards
        states = (states - states.mean()) / (states.std() + 1e-8)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Compute advantages
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            advantages = rewards + self.gamma * next_values - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.detach()
            value_target = rewards + self.gamma * next_values
            value_target = value_target.detach()

        # PPO update
        for _ in range(5):  # Multiple epochs
            # Actor update
            dist = self.actor(states)
            new_log_probs = dist.log_prob(actions).sum(1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic update
            value_pred = self.critic(states).squeeze()
            critic_loss = nn.MSELoss()(value_pred, value_target)
            
            # Combined loss
            total_loss = actor_loss + 0.5 * critic_loss
            
            # Gradient update with clipping
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save_checkpoint(self, episode, rewards):
        """Save a checkpoint of the model."""
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'rewards': rewards
        }
        path = self.checkpoint_dir / f'checkpoint_episode_{episode}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        return checkpoint['episode'], checkpoint['rewards']

    def save_model(self, path):
        """Save just the model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, path)

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
            action, log_prob = agent.select_actions(state)
            
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