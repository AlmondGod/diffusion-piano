"""
Improved PPO implementation with stability enhancements
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import wandb  # For logging training metrics

class RolloutBuffer:
    """Stores and processes rollout data"""
    def __init__(self, batch_size=64):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.next_states = []
        self.dones = []
        self.batch_size = batch_size
        
    def push(self, state, action, reward, log_prob, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.next_states.append(next_state)
        self.dones.append(done)
        
    def get(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.rewards)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(np.array(self.next_states)),
            torch.FloatTensor(np.array(self.dones))
        )
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.next_states.clear()
        self.dones.clear()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        # Initialize log_std with lower value for more precise initial actions
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)
        
    def forward(self, state):
        mean = self.network(state)
        # Clamp std for stability
        std = torch.exp(torch.clamp(self.log_std, -20, 2))
        return Normal(mean, std)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),  # Add dropout for regularization
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),  # Gradually reduce size
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights with smaller values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        return self.network(state)

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def __call__(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

        return (x - self.mean) / np.sqrt(self.var + 1e-8)

class PPOAgent:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        lr=1e-4,
        gamma=0.99,
        epsilon=0.2,
        entropy_coef=0.01,
        value_coef=1.0,
        max_grad_norm=0.5,
        ppo_epochs=10,
        batch_size=64,
        device='cuda',
        checkpoint_dir='checkpoints',
        use_wandb=True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        # Reduced learning rates and added beta parameters for Adam
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=lr*2,  # Reduced multiplier from 3 to 2
            betas=(0.9, 0.999),
            eps=1e-5
        )
        
        # More gradual learning rate decay
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer,
            mode='max',
            factor=0.5,
            patience=100,
            verbose=True
        )
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer,
            mode='min',
            factor=0.5,
            patience=100,
            verbose=True
        )
        
        # Add reward normalization
        self.reward_normalizer = RunningMeanStd()
        
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.rollout_buffer = RolloutBuffer(batch_size)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="robopianist-ppo", config={
                "lr": lr,
                "gamma": gamma,
                "epsilon": epsilon,
                "entropy_coef": entropy_coef,
                "value_coef": value_coef,
                "ppo_epochs": ppo_epochs,
                "batch_size": batch_size
            })

    def select_actions(self, states):
        """Handle batch of states with exploration noise"""
        states = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            dist = self.actor(states)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(1)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def update(self, states, actions, rewards, log_probs, next_states, dones):
        # Normalize rewards
        rewards = torch.FloatTensor(self.reward_normalizer(rewards)).to(self.device)
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute returns and advantages
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            
            # Calculate returns using TD(λ)
            returns = torch.zeros_like(rewards)
            future_returns = next_values * (1 - dones)  # Zero out future values for terminal states
            returns = rewards + self.gamma * future_returns
            
            # GAE Advantage calculation
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = next_values[t]
                else:
                    next_value = values[t + 1]
                    
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
                advantages[t] = gae

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.ppo_epochs):
            # Create data loader for mini-batch updates
            dataset = torch.utils.data.TensorDataset(
                states, actions, old_log_probs, advantages, returns
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            for batch in loader:
                b_states, b_actions, b_old_log_probs, b_advantages, b_returns = batch
                
                # Actor loss
                dist = self.actor(b_states)
                new_log_probs = dist.log_prob(b_actions).sum(1)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Critic loss with larger coefficient
                value_pred = self.critic(b_states).squeeze(-1)
                critic_loss = nn.MSELoss()(value_pred, b_returns)
                
                # Update critic first
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Then update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                if self.use_wandb:
                    wandb.log({
                        "actor_loss": actor_loss.item(),
                        "critic_loss": critic_loss.item(),
                        "entropy": entropy.item(),
                        "value_predictions": value_pred.mean().item(),
                        "returns": b_returns.mean().item(),
                        "advantages": b_advantages.mean().item()
                    })
        
        # Update learning rates based on performance
        mean_reward = rewards.mean().item()
        mean_critic_loss = critic_loss.item()
        self.actor_scheduler.step(mean_reward)
        self.critic_scheduler.step(mean_critic_loss)

    def save_checkpoint(self, episode, rewards):
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
            'rewards': rewards
        }
        path = self.checkpoint_dir / f'checkpoint_episode_{episode}.pt'
        torch.save(checkpoint, path)
        if self.use_wandb:
            wandb.save(str(path))
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
        return checkpoint['episode'], checkpoint['rewards']