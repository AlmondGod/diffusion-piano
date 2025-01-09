import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch
import argparse
from pathlib import Path
import time

from colab_train_ppo import PianoEnvWrapper
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file
from robopianist.wrappers import PianoSoundVideoWrapper
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
from mujoco_utils import composer_utils
from gymnasium import spaces
from typing import Dict

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
        self.env = PianoSoundVideoWrapper(
            self.env,
            record_every=1,
            camera_id="piano/back",
            record_dir=".",
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

def setup_inference():
    """Setup CUDA and PyTorch for inference"""
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("Using CPU")
        device = 'cpu'
    return device

def load_model(model_path: str, env, device: str) -> PPO:
    """Load the trained model"""
    model = PPO.load(
        model_path,
        env=env,
        device=device,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[1024, 1024, 1024, 512, 512],
                vf=[1024, 1024, 1024, 512, 512]
            ),
            activation_fn=torch.nn.ReLU,
            normalize_images=False,
        ),
    )
    return model

def run_inference(
    model: PPO,
    env: gym.Env,
    num_episodes: int = 1,
    render: bool = False,
    deterministic: bool = True,
) -> None:
    """Run inference on the trained model"""
    for episode in range(num_episodes):
        print(f"\nStarting episode {episode + 1}")
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if step % 100 == 0:
                print(f"Step {step}, Current reward: {reward:.3f}, Total reward: {total_reward:.3f}")
            step += 1
        
        print(f"Episode {episode + 1} finished after {step} steps")
        print(f"Total reward: {total_reward:.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/final_model.zip')
    parser.add_argument('--midi_path', type=str, default='midi_files_cut/Guren no Yumiya Cut 14s.mid')
    parser.add_argument('--fingering_path', type=str, default='data_processing/Guren no Yumiya Cut 14s_fingering v3.txt')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--render', action='store_true', help='Enable rendering', default=True)
    parser.add_argument('--non_deterministic', action='store_true', help='Use non-deterministic actions', default=False)
    args = parser.parse_args()

    # Setup device
    device = setup_inference()

    # Load MIDI sequence
    print("Loading MIDI sequence...")
    midi_sequence = add_fingering_from_annotation_file(
        args.midi_path,
        args.fingering_path
    )

    # Create environment
    print("Creating environment...")
    env = PianoEnvWrapper(midi_sequence)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, env, device)

    # Run inference
    print("Starting inference...")
    run_inference(
        model=model,
        env=env,
        num_episodes=args.num_episodes,
        render=args.render,
        deterministic=not args.non_deterministic
    )

if __name__ == "__main__":
    main()
