import numpy as np
from dm_env_wrappers import CanonicalSpecWrapper
from mujoco_utils import composer_utils
from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file
from ppo_base import PPOAgent
from pathlib import Path
import json
import argparse
import jax
import mujoco.mjx as mjx
import os

# Set XLA flags for better GPU performance
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=true'

class VectorizedPianoEnv:
    def __init__(self, num_envs, midi_sequence):
        self.num_envs = num_envs
        self.envs = []
        
        # Create a single task/environment first
        task = PianoWithShadowHands(
            midi=midi_sequence,
            n_steps_lookahead=1,
            trim_silence=True,
            wrong_press_termination=False,
            initial_buffer_time=0.0,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_colorization=False,
            disable_hand_collisions=False,
        )
        
        # Create MJX model from the task
        self.model = mjx.Device(task.physics.model)
        
        # Initialize states for all environments
        self.states = jax.vmap(self.model.make_state)(jax.random.split(jax.random.PRNGKey(0), num_envs))
        
        # Store observation/action specs from base environment
        base_env = composer_utils.Environment(
            task=task,
            strip_singleton_obs_buffer_dim=True,
            recompile_physics=True
        )
        self.base_env = CanonicalSpecWrapper(base_env)
        self.observation_spec = self.base_env.observation_spec()
        self.action_spec = self.base_env.action_spec()
        
    @jax.jit
    def step_batch(self, states, actions):
        """Perform one physics step for all environments in parallel"""
        next_states = jax.vmap(self.model.step)(states, actions)
        return next_states
    
    def reset(self):
        """Reset all environments"""
        # Reset states using MJX
        self.states = jax.vmap(self.model.make_state)(jax.random.split(jax.random.PRNGKey(0), self.num_envs))
        
        # Get observations from states
        observations = self._get_observations(self.states)
        return observations
    
    def step(self, actions):
        """Step all environments in parallel"""
        # Convert actions to device array
        actions = jax.device_put(actions)
        
        # Step physics in parallel
        self.states = self.step_batch(self.states, actions)
        
        # Get observations and rewards
        next_obs = self._get_observations(self.states)
        rewards = self._compute_rewards(self.states)
        dones = self._compute_dones(self.states)
        
        return next_obs, rewards, dones
    
    def _get_observations(self, states):
        """Extract observations from states for all environments"""
        # Implementation depends on what observations you need
        # This is a placeholder - implement based on your needs
        obs_dict = {}
        for key in self.observation_spec.keys():
            # Extract observation for each key from states
            obs_dict[key] = self._extract_observation(states, key)
        return obs_dict
    
    def _extract_observation(self, states, key):
        """Extract specific observation from states"""
        # Implement based on your observation space
        # This is a placeholder
        if key == 'goal':
            return jax.vmap(lambda s: s.qpos[:178])(states)
        elif key == 'fingering':
            return jax.vmap(lambda s: s.qpos[178:188])(states)
        # Add other observation extractions as needed
        return np.zeros((self.num_envs, self.observation_spec[key].shape[0]))
    
    def _compute_rewards(self, states):
        """Compute rewards for all environments"""
        # Implement your reward function
        # This is a placeholder
        return np.zeros(self.num_envs)
    
    def _compute_dones(self, states):
        """Compute done flags for all environments"""
        # Implement your termination conditions
        # This is a placeholder
        return np.zeros(self.num_envs, dtype=bool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_episodes', type=int, default=100)
    args = parser.parse_args()

    # Load MIDI sequence
    midi_sequence = add_fingering_from_annotation_file(
        "./midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "./data_processing/Guren no Yumiya Cut 14s_fingering v3.txt"
    )

    # Create vectorized environment
    num_envs = args.num_envs
    num_episodes = args.num_episodes
    vec_env = VectorizedPianoEnv(num_envs, midi_sequence)

    # Get dimensions from first environment
    observation_spec = vec_env.envs[0].observation_spec()
    action_spec = vec_env.envs[0].action_spec()

    state_dim = sum(np.prod(spec.shape) for spec in observation_spec.values())
    action_dim = action_spec.shape[0]

    print(f"State dimension: {state_dim}")  # Debug print
    print(f"Action dimension: {action_dim}")  # Debug print

    # Add these parameters
    checkpoint_dir = 'checkpoints'
    checkpoint_frequency = 10  # Save checkpoint every N episodes
    model_dir = 'models'
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Initialize the agent with checkpoint directory
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        device='cuda',
        checkpoint_dir=checkpoint_dir
    )

    # Training history
    history = {
        'episode_rewards': [],
        'mean_rewards': []
    }

    # Training loop
    for episode in range(num_episodes):
        obs = vec_env.reset()
        episode_rewards = np.zeros(num_envs)
        
        while True:
            # Stack and flatten observations from all environments
            states = np.stack([
                np.concatenate([
                    obs['goal'][i].flatten(),
                    obs['fingering'][i].flatten(),
                    obs['piano/state'][i].flatten(),
                    obs['piano/sustain_state'][i].flatten(),
                    obs['rh_shadow_hand/joints_pos'][i].flatten(),
                    obs['lh_shadow_hand/joints_pos'][i].flatten()
                ]) for i in range(num_envs)
            ])
            
            # Get actions for all environments
            actions, log_probs = agent.select_actions(states)
            
            # Step all environments
            next_obs, rewards, dones = vec_env.step(actions)
            
            # Flatten next_states the same way as states
            next_states = np.stack([
                np.concatenate([
                    next_obs['goal'][i].flatten(),
                    next_obs['fingering'][i].flatten(),
                    next_obs['piano/state'][i].flatten(),
                    next_obs['piano/sustain_state'][i].flatten(),
                    next_obs['rh_shadow_hand/joints_pos'][i].flatten(),
                    next_obs['lh_shadow_hand/joints_pos'][i].flatten()
                ]) for i in range(num_envs)
            ])
            
            episode_rewards += rewards
            
            # Update agent with batch of experiences
            agent.update(
                states=states,
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
                next_states=next_states
            )
            
            if all(dones):
                break
                
            obs = next_obs
        
        mean_reward = episode_rewards.mean()
        print(f"Episode {episode}, Mean Reward: {mean_reward}")
        
        # Save history
        history['episode_rewards'].append(episode_rewards.tolist())
        history['mean_rewards'].append(mean_reward)
        
        # Save checkpoint periodically
        if episode > 0 and episode % checkpoint_frequency == 0:
            agent.save_checkpoint(episode, history)

    # Save final model and training history
    final_model_path = Path(model_dir) / 'final_model.pt'
    agent.save_model(final_model_path)

    # Save training history
    history_path = Path(model_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)

    print(f"Training complete. Model saved to {final_model_path}")
    print(f"Training history saved to {history_path}")

