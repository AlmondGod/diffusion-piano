import numpy as np
from dm_env_wrappers import CanonicalSpecWrapper
from mujoco_utils import composer_utils
from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
import numpy as np
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file
from ppo_v2 import PPOAgent
from pathlib import Path
import json
import argparse
import time

# Instead, load the midi file then add the fingering annotations to it as a sequence,
# then convert the sequence to a midi_file object

midi_sequence = add_fingering_from_annotation_file(
        "/Users/almondgod/Repositories/robopianist/midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "/Users/almondgod/Repositories/robopianist/data_processing/Guren no Yumiya Cut 14s_fingering v3.txt"
    )

class VectorizedPianoEnv:
    def __init__(self, num_envs, midi_sequence):
        self.num_envs = num_envs
        self.envs = []
        
        for _ in range(num_envs):
            # Create task
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
            env = composer_utils.Environment(
                task=task, 
                strip_singleton_obs_buffer_dim=True, 
                recompile_physics=True
            )
            env = CanonicalSpecWrapper(env)
            
            self.envs.append(env)
            
    def reset(self):
        """Reset all environments"""
        observations = [env.reset().observation for env in self.envs]
        return self._stack_obs(observations)
    
    def step(self, actions):
        """Step all environments in parallel"""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        next_obs = [timestep.observation for timestep in results]
        rewards = [timestep.reward for timestep in results]
        dones = [timestep.last() for timestep in results]
        
        return self._stack_obs(next_obs), np.array(rewards), np.array(dones)
    
    def _stack_obs(self, observations):
        """Stack observations from all environments"""
        stacked_obs = {}
        for key in observations[0].keys():
            stacked_obs[key] = np.stack([obs[key] for obs in observations])
        return stacked_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--num_episodes', type=int, default=4000)
    args = parser.parse_args()

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
    checkpoint_dir = f'checkpoints/{time.strftime("%Y%m%d_%H%M%S")}'
    checkpoint_frequency = 200  # Save checkpoint every N episodes
    model_dir = f'models/{time.strftime("%Y%m%d_%H%M%S")}'
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Initialize the agent with checkpoint directory
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        initial_entropy_coef=0.1,
        min_entropy_coef=0.01,
        entropy_decay_steps=10000,  # Faster decay for 4000 episodes
        batch_size=128,  # Increased from 64
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
                next_states=next_states,
                dones=dones
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

    # Save final model and training history demarked by time
    final_model_path = Path(model_dir) / f'final_model_{time.strftime("%Y%m%d_%H%M%S")}.pt'
    agent.save_model(final_model_path)

    # Save training history
    history_path = Path(model_dir) / f'training_history_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)

    print(f"Training complete. Model saved to {final_model_path}")
    print(f"Training history saved to {history_path}")