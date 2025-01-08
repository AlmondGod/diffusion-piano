import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from typing import Dict, Any

import os
from dm_env_wrappers import CanonicalSpecWrapper
from mujoco_utils import composer_utils
from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file
from pathlib import Path
import argparse
import time

import torch
from stable_baselines3.common.callbacks import BaseCallback
import psutil
from torch.cuda import amp

def setup_monitoring():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory total: {info.total / 1024**2:.2f} MB")
        print(f"GPU memory used: {info.used / 1024**2:.2f} MB")
    except:
        print("Could not initialize GPU monitoring")

    # Enable PyTorch optimizations
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    # Print CPU info
    import multiprocessing
    print(f"Number of CPU cores: {multiprocessing.cpu_count()}")

class ResourceMonitorCallback(BaseCallback):
    """
    Custom callback for monitoring system resources during training.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_start = time.time()
        
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:  # Log every 1000 steps
            # GPU Memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0)/1024**2
                reserved = torch.cuda.memory_reserved(0)/1024**2
                print(f"\nStep {self.n_calls}")
                print(f"GPU Memory allocated: {allocated:.2f}MB")
                print(f"GPU Memory reserved: {reserved:.2f}MB")
            
            # CPU Usage
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            print(f"CPU Usage: {cpu_percent}%")
            print(f"RAM Usage: {ram_percent}%")
            
            # Training time
            elapsed_time = time.time() - self.training_start
            print(f"Training time: {elapsed_time/3600:.2f} hours")
            
        return True

class PianoEnvWrapper(gym.Env):
    def __init__(self, midi_sequence):
        super().__init__()
        
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
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--total_timesteps', type=int, default=2000000)
    parser.add_argument('--checkpoint_freq', type=int, default=10000)
    parser.add_argument('--eval_freq', type=int, default=20000)
    args = parser.parse_args()

    # Setup monitoring
    setup_monitoring()
    
    # Create output directories with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = f'training_runs/{timestamp}'
    model_dir = f'{base_dir}/models'
    log_dir = f'{base_dir}/logs'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    midi_sequence = add_fingering_from_annotation_file(
        "./midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "./data_processing/Guren no Yumiya Cut 14s_fingering v3.txt"
    )

    # Create vectorized environment
    env_fns = [make_env(midi_sequence, i) for i in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=model_dir,
        name_prefix="ppo_piano"
    )
    
    monitor_callback = ResourceMonitorCallback()
    
    # Initialize PPO with mixed precision training
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.1,
        verbose=1,
        device='cuda',
        policy_kwargs=dict(
            net_arch=dict(
                pi=[1024, 1024, 1024, 512, 512],
                vf=[1024, 1024, 1024, 512, 512]
            ),
            activation_fn=torch.nn.ReLU,
            normalize_images=False,
        ),
        tensorboard_log="./piano_tensorboard/"
    )

    # Instead of manually converting to fp16, use torch.cuda.amp
    model.policy.optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    scaler = amp.GradScaler('cuda')
    
    # Add gradient clipping
    model.policy.optimizer.param_groups[0]['grad_clip_norm'] = 0.5

    try:
        # Train with callbacks
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, monitor_callback],
        )
        
        # Save final model
        final_model_path = f"{model_dir}/final_model"
        model.save(final_model_path)
        print(f"Training complete. Model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")
        model.save(f"{model_dir}/interrupted_model")
        print("Model saved. Exiting...")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        model.save(f"{model_dir}/error_model")
        raise e