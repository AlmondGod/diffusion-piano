import argparse
import numpy as np
import torch
from stable_baselines3 import SAC
from doubleq_policy import DroQPolicy
from colab_play_ppo import PianoEnvWrapper, setup_inference
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                      help='Path to trained model checkpoint', default='/Users/almondgod/Repositories/robopianist/models/droq_piano_1600000_steps.zip')
    parser.add_argument('--midi_path', type=str, 
                      help='Path to MIDI file', default='./midi_files_cut/Guren no Yumiya Cut 14s.mid')
    parser.add_argument('--fingering_path', type=str,
                      help='Path to fingering annotation file', default='./data_processing/Guren no Yumiya Cut 14s_fingering v3.txt')
    parser.add_argument('--deterministic', action='store_true',
                      help='Use deterministic actions')
    return parser.parse_args()

def run_inference(model: SAC, env: PianoEnvWrapper, deterministic: bool = True) -> None:
    """Run one episode of inference with recording"""
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
    
    print(f"\nEpisode finished after {step} steps")
    print(f"Total reward: {total_reward:.3f}")

def main():
    args = parse_args()
    device = setup_inference()

    # Load MIDI sequence
    print("Loading MIDI sequence...")
    midi_sequence = add_fingering_from_annotation_file(
        args.midi_path,
        args.fingering_path
    )

    # Create environment with recording wrapper
    print("Creating environment...")
    env = PianoEnvWrapper(midi_sequence)

    # Load model without specifying policy_kwargs
    print(f"Loading model from {args.model_path}...")
    model = SAC.load(
        args.model_path,
        env=env,
        device=device
    )

    # Run inference
    print("Starting inference...")
    run_inference(model, env, deterministic=args.deterministic)

if __name__ == "__main__":
    main() 