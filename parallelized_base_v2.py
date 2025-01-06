import numpy as np
from dm_env_wrappers import CanonicalSpecWrapper
from mujoco_utils import composer_utils
from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file
from ppo_base import PPOAgent
from pathlib import Path
import json
import argparse
from acme import wrappers
from acme.environments import base
import dm_env

class PianoEnvFactory:
    """Factory for creating piano environments"""
    def __init__(self, midi_sequence):
        self.midi_sequence = midi_sequence
    
    def __call__(self) -> dm_env.Environment:
        task = PianoWithShadowHands(
            midi=self.midi_sequence,
            n_steps_lookahead=1,
            trim_silence=True,
            wrong_press_termination=False,
            initial_buffer_time=0.0,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_colorization=False,
            disable_hand_collisions=False,
        )
        env = composer_utils.Environment(
            task=task, 
            strip_singleton_obs_buffer_dim=True, 
            recompile_physics=True
        )
        return CanonicalSpecWrapper(env)

def create_parallel_envs(num_envs: int, midi_sequence) -> base.EnvironmentWrapper:
    """Create parallel environment wrapper"""
    env_factory = PianoEnvFactory(midi_sequence)
    return wrappers.ParallelEnvironmentWrapper(
        env_factory,
        num_parallel_environments=num_envs,
        batch_size=num_envs,
        start_method='spawn'  # Important for CUDA support
    )

def process_observation(obs, num_envs):
    """Process and flatten observation dictionary"""
    return np.stack([
        np.concatenate([
            obs['goal'][i].flatten(),
            obs['fingering'][i].flatten(),
            obs['piano/state'][i].flatten(),
            obs['piano/sustain_state'][i].flatten(),
            obs['rh_shadow_hand/joints_pos'][i].flatten(),
            obs['lh_shadow_hand/joints_pos'][i].flatten()
        ]) for i in range(num_envs)
    ])

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_episodes', type=int, default=100)
    args = parser.parse_args()

    # Load MIDI sequence
    midi_sequence = add_fingering_from_annotation_file(
        "./midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "./data_processing/Guren no Yumiya Cut 14s_fingering v3.txt"
    )

    # Create parallel environment
    env = create_parallel_envs(args.num_envs, midi_sequence)

    # Get specs from environment
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    state_dim = sum(np.prod(spec.shape) for spec in observation_spec.values())
    action_dim = action_spec.shape[0]

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Setup directories
    checkpoint_dir = 'checkpoints'
    checkpoint_frequency = 10
    model_dir = 'models'
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Initialize agent
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
    for episode in range(args.num_episodes):
        timestep = env.reset()
        episode_rewards = np.zeros(args.num_envs)
        
        while not timestep.last():
            # Process observations
            states = process_observation(timestep.observation, args.num_envs)
            
            # Get actions
            actions, log_probs = agent.select_actions(states)
            
            # Step environment
            timestep = env.step(actions)
            
            # Process next states
            next_states = process_observation(timestep.observation, args.num_envs)
            
            # Update rewards
            episode_rewards += timestep.reward
            
            # Update agent
            agent.update(
                states=states,
                actions=actions,
                rewards=timestep.reward,
                log_probs=log_probs,
                next_states=next_states
            )
        
        # Process episode results
        mean_reward = episode_rewards.mean()
        print(f"Episode {episode}, Mean Reward: {mean_reward}")
        
        # Save history
        history['episode_rewards'].append(episode_rewards.tolist())
        history['mean_rewards'].append(mean_reward)
        
        # Save checkpoint
        if episode > 0 and episode % checkpoint_frequency == 0:
            agent.save_checkpoint(episode, history)

    # Save final results
    final_model_path = Path(model_dir) / 'final_model.pt'
    agent.save_model(final_model_path)

    history_path = Path(model_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)

    print(f"Training complete. Model saved to {final_model_path}")
    print(f"Training history saved to {history_path}")