import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch
import argparse
from pathlib import Path
import time

from colab_train_ppo import PianoEnvWrapper
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file

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
