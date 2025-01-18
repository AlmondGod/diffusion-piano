import dataclasses
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import madrona_mjx
from madrona_mjx import MadronaMJX, MadronaConfig

@dataclasses.dataclass
class RoboPianistConfig:
    """Configuration for RoboPianist environment."""
    num_worlds: int = 1024
    episode_length: int = 1000
    ctrl_dt: float = 0.02
    render_width: int = 64
    render_height: int = 64
    camera_id: str = "piano/back"

class RoboPianistMadrona:
    """GPU-accelerated RoboPianist using Madrona-MJX."""
    
    def __init__(self, midi_sequence, config: Optional[RoboPianistConfig] = None):
        self.config = config or RoboPianistConfig()
        
        # Create base task
        from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
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

        # Convert to MJX and batch
        self.model = mjx.Device.init(self.task.env.physics.model)
        
        # Initialize Madrona renderer
        madrona_config = MadronaConfig(
            num_worlds=self.config.num_worlds,
            render_width=self.config.render_width,
            render_height=self.config.render_height,
            use_raytracing=True,  # Use raytracer backend
        )
        
        self.renderer = MadronaMJX(
            model=self.model,
            config=madrona_config,
            camera_id=self.config.camera_id,
        )
        
        # Cache dimensions
        self._obs_dim = sum(jnp.prod(jnp.array(spec.shape)) 
                           for spec in self.task.env.observation_spec().values())
        self._act_dim = self.task.env.action_spec().shape[0]
        
    def reset(self, rng: jnp.ndarray) -> Dict:
        """Reset all environments."""
        # Initialize states for all environments in parallel
        keys = jax.random.split(rng, self.config.num_worlds)
        states = jax.vmap(self.model.reset)(keys)
        
        # Get observations including rendered images
        obs = self._get_obs(states)
        
        return {
            'physics': states,
            'obs': obs,
            'reward': jnp.zeros(self.config.num_worlds),
            'done': jnp.zeros(self.config.num_worlds, dtype=bool),
            'info': {'time': jnp.zeros(self.config.num_worlds)}
        }

    def step(self, state: Dict, action: jnp.ndarray) -> Dict:
        """Step all environments in parallel."""
        # Step physics
        next_physics = jax.vmap(self.model.step)(state['physics'], action)
        
        # Get observations and rewards
        next_obs = self._get_obs(next_physics)
        rewards = self._compute_reward(next_physics)
        dones = (state['info']['time'] >= self.config.episode_length)
        
        return {
            'physics': next_physics,
            'obs': next_obs,
            'reward': rewards,
            'done': dones,
            'info': {'time': state['info']['time'] + 1}
        }

    def _get_obs(self, physics) -> Dict:
        """Get observations including rendered images."""
        # Get standard observations
        obs = {
            'goal': physics.data.qpos[self.task._piano_keys_qpos_indices],
            'fingering': self.task._current_fingering,
            'piano/state': physics.data.qpos[self.task._piano_keys_qpos_indices],
            'piano/sustain_state': jnp.zeros(1),
            'rh_shadow_hand/joints_pos': physics.data.qpos[self.task._rh_qpos_indices],
            'lh_shadow_hand/joints_pos': physics.data.qpos[self.task._lh_qpos_indices],
        }
        
        # Add rendered observations
        rgb, depth = self.renderer.render(physics)
        obs.update({
            'rgb': rgb,
            'depth': depth,
        })
        
        return obs

    def _compute_reward(self, physics) -> jnp.ndarray:
        """Compute rewards for all environments."""
        pressed_keys = physics.data.qpos[self.task._piano_keys_qpos_indices]
        target_keys = self.task._current_fingering
        return -jnp.mean(jnp.abs(pressed_keys - target_keys))

    @property
    def dt(self) -> float:
        return self.config.ctrl_dt 