import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any

class RoboPianistMJX:
    """Simplified RoboPianist environment using MJX for GPU-accelerated training"""
    
    def __init__(self, midi_sequence, num_envs=1024):
        # Load the MuJoCo model from the original environment
        from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
        
        # Create task without sound/video
        self.task = PianoWithShadowHands(
            midi=midi_sequence,
            n_steps_lookahead=1,
            trim_silence=True,
            wrong_press_termination=False,
            initial_buffer_time=0.0,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_hand_collisions=False,
        )
        
        # Convert to MJX
        self.model = mjx.Device.init(self.task.env.physics.model)
        self.num_envs = num_envs
        
        # Cache observation/action dimensions
        self._obs_dim = sum(np.prod(spec.shape) for spec in self.task.env.observation_spec().values())
        self._act_dim = self.task.env.action_spec().shape[0]
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.task.env.action_spec().minimum,
            high=self.task.env.action_spec().maximum,
            shape=self.task.env.action_spec().shape,
            dtype=np.float32
        )
    
    def _process_obs(self, physics) -> jp.ndarray:
        """Convert MuJoCo state to flat observation array"""
        obs_dict = {
            'goal': physics.data.qpos[self.task._piano_keys_qpos_indices],
            'fingering': self.task._current_fingering,
            'piano/state': physics.data.qpos[self.task._piano_keys_qpos_indices],
            'piano/sustain_state': jp.zeros(1),  # Simplified - no sustain
            'rh_shadow_hand/joints_pos': physics.data.qpos[self.task._rh_qpos_indices],
            'lh_shadow_hand/joints_pos': physics.data.qpos[self.task._lh_qpos_indices]
        }
        return jp.concatenate([v.flatten() for v in obs_dict.values()])

    def reset(self, rng: jp.ndarray) -> Dict[str, Any]:
        """Reset all environments"""
        # Initialize states for all environments in parallel
        keys = jax.random.split(rng, self.num_envs)
        states = jax.vmap(self.model.reset)(keys)
        
        # Get observations
        obs = jax.vmap(self._process_obs)(states)
        
        return {
            'physics': states,
            'obs': obs,
            'reward': jp.zeros(self.num_envs),
            'done': jp.zeros(self.num_envs, dtype=bool)
        }

    def step(self, state: Dict[str, Any], action: jp.ndarray) -> Dict[str, Any]:
        """Step all environments in parallel"""
        # Step physics
        next_physics = jax.vmap(self.model.step)(state['physics'], action)
        
        # Get observations and rewards
        next_obs = jax.vmap(self._process_obs)(next_physics)
        rewards = self._compute_reward(next_physics)
        dones = self._compute_done(next_physics)
        
        return {
            'physics': next_physics,
            'obs': next_obs,
            'reward': rewards,
            'done': dones
        }

    def _compute_reward(self, physics) -> jp.ndarray:
        """Compute rewards for all environments"""
        # Simplified reward function - just key press accuracy
        pressed_keys = physics.data.qpos[self.task._piano_keys_qpos_indices]
        target_keys = self.task._current_fingering  # This needs to be tracked properly
        return -jp.mean(jp.abs(pressed_keys - target_keys))

    def _compute_done(self, physics) -> jp.ndarray:
        """Compute done flags for all environments"""
        # Simplified termination condition
        return jp.zeros(self.num_envs, dtype=bool)  # Fixed episode length

    @property
    def dt(self):
        return self.model.opt.timestep 