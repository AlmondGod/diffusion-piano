import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from typing import Dict, Any

from dm_env_wrappers import CanonicalSpecWrapper
from mujoco_utils import composer_utils
from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file
from pathlib import Path
import argparse
import time

import torch


class PianoEnvWrapper(gym.Env):
    def __init__(self, midi_sequence):
        super().__init__()
        
        # Create the base environment
        self.task = PianoWithShadowHands(
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
        self.env = composer_utils.Environment(
            task=self.task,
            strip_singleton_obs_buffer_dim=True,
            recompile_physics=True
        )
        self.env = CanonicalSpecWrapper(self.env)
        
        # Get specs from the environment
        obs_spec = self.env.observation_spec()
        action_spec = self.env.action_spec()
        
        # Calculate observation space size
        self.obs_dim = sum(np.prod(spec.shape) for spec in obs_spec.values())
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            shape=action_spec.shape,
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

    def _process_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert observation dict to flat array"""
        return np.concatenate([
            obs['goal'].flatten(),
            obs['fingering'].flatten(),
            obs['piano/state'].flatten(),
            obs['piano/sustain_state'].flatten(),
            obs['rh_shadow_hand/joints_pos'].flatten(),
            obs['lh_shadow_hand/joints_pos'].flatten()
        ])

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        timestep = self.env.reset()
        return self._process_obs(timestep.observation), {}

    def step(self, action):
        timestep = self.env.step(action)
        obs = self._process_obs(timestep.observation)
        reward = timestep.reward if timestep.reward is not None else 0.0
        terminated = timestep.last()
        truncated = False
        return obs, reward, terminated, truncated, {}

def make_env(midi_sequence, rank):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = PianoEnvWrapper(midi_sequence)
        return env
    return _init


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--total_timesteps', type=int, default=2000000)
    args = parser.parse_args()

    midi_sequence = add_fingering_from_annotation_file(
        "./midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "./data_processing/Guren no Yumiya Cut 14s_fingering v3.txt"
    )

    # Create vectorized environment
    env_fns = [make_env(midi_sequence, i) for i in range(args.num_envs)]  # 16 parallel envs
    vec_env = SubprocVecEnv(env_fns)

    # Set up model saving
    model_dir = f'models/{time.strftime("%Y%m%d_%H%M%S")}'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="ppo_piano"
    )
    

    # Initialize PPO with A100-optimized parameters
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=4096,          # Increased steps per update
        batch_size=256,        # Larger batches
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.1,
        verbose=1,
        device='cuda',
        # A100-specific optimizations
        policy_kwargs=dict(
            net_arch=dict(
                pi=[1024, 1024, 512, 512, 256],  # Policy network
                vf=[1024, 1024, 512, 512, 256]  # Value network
            ),
            activation_fn=torch.nn.ReLU,
            normalize_images=False,
        ),
        # Tensorboard logging
        tensorboard_log="./piano_tensorboard/"
    )

    # Add gradient clipping for stability with larger batches
    model.policy.optimizer.param_groups[0]['grad_clip_norm'] = 0.5

    # Train with larger total timesteps
    model.learn(
        total_timesteps=args.total_timesteps,  # Increased total timesteps
        callback=checkpoint_callback,
    )

    # Save final model
    final_model_path = f"{model_dir}/final_model"
    model.save(final_model_path)
    print(f"Training complete. Model saved to {final_model_path}")