import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Tuple, Type, Union
import numpy as np
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback

class LayerNormMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        dropout_rate: float = 0.01
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class CombinedCritic(nn.Module):
    def __init__(self, q1, q2):
        super().__init__()
        self.q1 = q1
        self.q2 = q2
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(torch.cat([obs, actions], dim=-1)), self.q2(torch.cat([obs, actions], dim=-1))

class ActorNetwork(nn.Module):
    def __init__(self, base_network):
        super().__init__()
        self.base_network = base_network
        self.dtype = torch.float32
        
    def forward(self, obs):
        return self.base_network(obs)
    
    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(dtype=self.dtype)
        mean, log_std = self.base_network(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Sum before the squashing correction to match dimensions
        log_prob = log_prob.sum(-1)
        # Add squashing correction
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)
        
        return action, log_prob

class DroQPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: callable,
        hidden_dims: List[int] = [256, 256, 256],
        dropout_rate: float = 0.01,
        target_entropy: float = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor_class=None,
            features_extractor_kwargs=None,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(eps=1e-5),
            squash_output=True,
        )
        
        # Ensure all tensors are float32
        self.dtype = torch.float32
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        
        # Q networks (double Q-learning)
        self.q1 = LayerNormMLP(obs_dim + action_dim, 1, hidden_dims, dropout_rate).to(dtype=self.dtype)
        self.q2 = LayerNormMLP(obs_dim + action_dim, 1, hidden_dims, dropout_rate).to(dtype=self.dtype)
        self.q1_target = LayerNormMLP(obs_dim + action_dim, 1, hidden_dims, dropout_rate).to(dtype=self.dtype)
        self.q2_target = LayerNormMLP(obs_dim + action_dim, 1, hidden_dims, dropout_rate).to(dtype=self.dtype)
        
        # Copy parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.critic = CombinedCritic(self.q1, self.q2)
        self.critic_target = CombinedCritic(self.q1_target, self.q2_target)

        # Actor network (tanh-diagonal-Gaussian)
        base_network = LayerNormMLP(obs_dim, action_dim * 2, hidden_dims, dropout_rate).to(dtype=self.dtype)
        self.actor = ActorNetwork(base_network)
        
        # Create optimizers and attach them to networks
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        
        # Create critic optimizer
        self.critic = CombinedCritic(self.q1, self.q2)
        self.critic.optimizer = self.optimizer_class(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )
        
        self.critic_target = CombinedCritic(self.q1_target, self.q2_target)
        
        # Automatic entropy tuning
        if target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(target_entropy)
            
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device, dtype=self.dtype)
        self.alpha = self.log_alpha.exp()
        
        self.tau = tau
        self.gamma = gamma
        self.q_values = []
        self.policy_losses = []

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Ensure input is float32
        observation = observation.to(dtype=self.dtype)
        mean, log_std = self.actor(observation).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        
        if deterministic:
            return torch.tanh(mean)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Compute log probability for logging
        log_prob = normal.log_prob(x_t).sum(axis=-1)
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)
        self.policy_losses.append(log_prob.mean().item())
        
        return action

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(obs, deterministic)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        # Get pre-tanh value (atanh(action))
        x_t = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(x_t)
        
        # Account for tanh squashing
        log_prob -= torch.sum(torch.log(1 - actions.pow(2) + 1e-6), dim=-1)
        log_prob = log_prob.sum(dim=-1)
        
        # Q-values
        q1 = self.q1(torch.cat([obs, actions], dim=-1))
        q2 = self.q2(torch.cat([obs, actions], dim=-1))
        
        # Log Q-values for monitoring
        self.q_values.append(torch.min(q1, q2).mean().item())
        
        return torch.min(q1, q2), log_prob, self.alpha.detach() * log_prob

    def update_target_networks(self) -> None:
        """Soft update of target networks"""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure input is float32
        obs = obs.to(dtype=self.dtype)
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Account for tanh squashing
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)
        log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.q_values = []
        self.policy_losses = []
    
    def _on_step(self) -> bool:
        if hasattr(self.model.policy, 'update_target_networks'):
            self.model.policy.update_target_networks()
            
        # Log metrics
        if hasattr(self.model.policy, 'q_values') and len(self.model.policy.q_values) > 0:
            self.logger.record("train/q_value", np.mean(self.model.policy.q_values))
            self.model.policy.q_values = []
        
        if hasattr(self.model.policy, 'policy_losses') and len(self.model.policy.policy_losses) > 0:
            self.logger.record("train/policy_loss", np.mean(self.model.policy.policy_losses))
            self.model.policy.policy_losses = []
            
        return True