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
from functools import partial
import mujoco

# Set XLA flags for better GPU performance
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=true'

class VectorizedPianoEnv:
    def __init__(self, num_envs, midi_sequence):
        self.num_envs = num_envs
        
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
            disable_hand_collisions=False,  # We can enable collisions now
        )

        # Create base environment to get model
        base_env = composer_utils.Environment(
            task=task,
            strip_singleton_obs_buffer_dim=True,
            recompile_physics=True
        )
        base_env = CanonicalSpecWrapper(base_env)
        
        # Get MuJoCo model and optimize for MJX
        mj_model = base_env.physics.model
        
        # Convert cylinders to capsules (supported type)
        for i in range(mj_model.ngeom):
            if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                # Get cylinder properties
                size = mj_model.geom_size[i].copy()  # [radius, height]
                pos = mj_model.geom_pos[i].copy()
                quat = mj_model.geom_quat[i].copy()
                
                # Convert to capsule
                mj_model.geom_type[i] = mujoco.mjtGeom.mjGEOM_CAPSULE
                # Capsule size: [radius, half-length]
                mj_model.geom_size[i] = np.array([size[0], size[1]/2])
                mj_model.geom_pos[i] = pos
                mj_model.geom_quat[i] = quat
        
        # Optimize model parameters for MJX
        mj_model.opt.iterations = 5  # Reduce solver iterations
        mj_model.opt.ls_iterations = 2  # Reduce line search iterations
        mj_model.opt.jacobian = 2  # Better for GPU (2 = dense)
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP  # Disable euler damping
        
        # Create MJX model and states
        self.mjx_model = mjx.put_model(mj_model)
        self.states = jax.vmap(self.mjx_model.make_state)(
            jax.random.split(jax.random.PRNGKey(0), num_envs)
        )
        
        # Store specs
        self.observation_spec = base_env.observation_spec()
        self.action_spec = base_env.action_spec()
        
    @partial(jax.jit, static_argnums=(0,))
    def step_batch(self, states, actions):
        """Perform one physics step for all environments in parallel"""
        return jax.vmap(mjx.step)(self.mjx_model, states, actions)
        
    def reset(self):
        """Reset all environments"""
        self.states = jax.vmap(self.mjx_model.make_state)(
            jax.random.split(jax.random.PRNGKey(0), self.num_envs)
        )
        return self._get_observations(self.states)
        
    def step(self, actions):
        """Step all environments in parallel"""
        actions = jax.device_put(actions)
        self.states = self.step_batch(self.states, actions)
        
        obs = self._get_observations(self.states) 
        rewards = self._compute_rewards(self.states)
        dones = self._compute_dones(self.states)
        
        return obs, rewards, dones

    def _get_observations(self, states):
        """Extract observations from states for all environments."""
        obs_dict = {}
        
        # Get qpos and qvel from states
        qpos = states.qpos  # Shape: (num_envs, nq)
        qvel = states.qvel  # Shape: (num_envs, nv)
        
        # Extract observations based on the original environment structure
        obs_dict['goal'] = qpos[:, :178]  # First 178 elements are goal positions
        obs_dict['fingering'] = qpos[:, 178:188]  # Next 10 elements are fingering
        obs_dict['piano/state'] = qpos[:, 188:276]  # 88 piano key states
        obs_dict['piano/sustain_state'] = qpos[:, 276:277]  # Sustain pedal state
        obs_dict['rh_shadow_hand/joints_pos'] = qpos[:, 277:303]  # Right hand joint positions
        obs_dict['lh_shadow_hand/joints_pos'] = qpos[:, 303:]  # Left hand joint positions
        
        return obs_dict

    def _compute_rewards(self, states):
        """Compute rewards for all environments."""
        # Get relevant state information
        qpos = states.qpos
        qvel = states.qvel
        
        # Extract piano key states and goal states
        piano_states = qpos[:, 188:276]  # 88 piano key states
        goal_states = qpos[:, :178]  # Goal positions
        
        # Compute key press accuracy
        key_accuracy = jax.numpy.sum(
            jax.numpy.abs(piano_states - goal_states[:, :88]), axis=1
        )
        
        # Compute velocity penalty to encourage smooth movements
        velocity_penalty = 0.1 * jax.numpy.sum(jax.numpy.square(qvel), axis=1)
        
        # Compute fingering reward
        fingering_states = qpos[:, 178:188]
        fingering_reward = 0.5 * jax.numpy.sum(
            jax.numpy.square(fingering_states), axis=1
        )
        
        # Combine rewards
        rewards = -key_accuracy - velocity_penalty + fingering_reward
        
        return rewards

    def _compute_dones(self, states):
        """Compute done flags for all environments."""
        qpos = states.qpos
        
        # Get piano key states and goal states
        piano_states = qpos[:, 188:276]
        goal_states = qpos[:, :178]
        
        # Episode is done if:
        # 1. Keys are significantly misaligned with goals
        key_error = jax.numpy.sum(
            jax.numpy.abs(piano_states - goal_states[:, :88]), axis=1
        )
        key_failure = key_error > 10.0
        
        # 2. Hands are in invalid positions (e.g., too far from piano)
        hand_positions = qpos[:, 277:]  # Both hands' joint positions
        hand_invalid = jax.numpy.any(
            jax.numpy.abs(hand_positions) > 2.0, axis=1
        )
        
        # Combine termination conditions
        dones = key_failure | hand_invalid
        
        return dones

    @partial(jax.jit, static_argnums=(0,))
    def _extract_observation(self, states, key):
        """Extract specific observation from states."""
        if key == 'goal':
            return states.qpos[:, :178]
        elif key == 'fingering':
            return states.qpos[:, 178:188]
        elif key == 'piano/state':
            return states.qpos[:, 188:276]
        elif key == 'piano/sustain_state':
            return states.qpos[:, 276:277]
        elif key == 'rh_shadow_hand/joints_pos':
            return states.qpos[:, 277:303]
        elif key == 'lh_shadow_hand/joints_pos':
            return states.qpos[:, 303:]
        else:
            raise KeyError(f"Unknown observation key: {key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=8192)
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

    # Get dimensions from observation and action specs
    state_dim = sum(np.prod(spec.shape) for spec in vec_env.observation_spec.values())
    action_dim = vec_env.action_spec.shape[0]

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Setup training
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

    history = {
        'episode_rewards': [],
        'mean_rewards': []
    }

    # Training loop
    @jax.jit
    def process_observations(obs):
        """Process observations using JAX operations."""
        return jax.numpy.stack([
            jax.numpy.concatenate([
                obs['goal'][i].flatten(),
                obs['fingering'][i].flatten(),
                obs['piano/state'][i].flatten(),
                obs['piano/sustain_state'][i].flatten(),
                obs['rh_shadow_hand/joints_pos'][i].flatten(),
                obs['lh_shadow_hand/joints_pos'][i].flatten()
            ]) for i in range(num_envs)
        ])

    for episode in range(num_episodes):
        obs = vec_env.reset()
        episode_rewards = jax.numpy.zeros(num_envs)
        
        while True:
            # Process observations using JAX
            states = process_observations(obs)
            
            # Get actions
            actions, log_probs = agent.select_actions(states)
            
            # Step environments
            next_obs, rewards, dones = vec_env.step(actions)
            
            # Process next observations
            next_states = process_observations(next_obs)
            
            # Update rewards
            episode_rewards += rewards
            
            # Update agent
            agent.update(
                states=states,
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
                next_states=next_states
            )
            
            if jax.numpy.all(dones):
                break
                
            obs = next_obs
        
        # Convert to numpy for logging
        mean_reward = float(episode_rewards.mean())
        print(f"Episode {episode}, Mean Reward: {mean_reward}")
        
        # Save history
        history['episode_rewards'].append(jax.device_get(episode_rewards).tolist())
        history['mean_rewards'].append(mean_reward)
        
        # Save checkpoint
        if episode > 0 and episode % checkpoint_frequency == 0:
            agent.save_checkpoint(episode, history)

    # Save final model and history
    final_model_path = Path(model_dir) / 'final_model.pt'
    agent.save_model(final_model_path)

    history_path = Path(model_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)

    print(f"Training complete. Model saved to {final_model_path}")
    print(f"Training history saved to {history_path}")

