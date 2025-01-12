import argparse
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from doubleq_policy import DroQPolicy, CustomCallback
from colab_train_ppo import PianoEnvWrapper, setup_monitoring, ResourceMonitorCallback

from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=8,
                      help='Number of parallel environments')
    parser.add_argument('--total_timesteps', type=int, default=5_000_000,
                      help='Total timesteps for training')
    parser.add_argument('--checkpoint_freq', type=int, default=10000,
                      help='Frequency of saving checkpoints')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str,
                      help='Path to checkpoint for resuming training')
    return parser.parse_args()

def main():
    args = parse_args()
    setup_monitoring()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup directories
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_dir = f'training_runs/{timestamp}'
    model_dir = f'{base_dir}/models'
    log_dir = f'{base_dir}/logs'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Load MIDI sequence
    midi_sequence = add_fingering_from_annotation_file(
        "./midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "./data_processing/Guren no Yumiya Cut 14s_fingering v3.txt"
    )

    # Create vectorized environment
    env_fns = [lambda: PianoEnvWrapper(midi_sequence) for _ in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=model_dir,
        name_prefix="droq_piano"
    )
    monitor_callback = ResourceMonitorCallback()
    droq_callback = CustomCallback()  # For target network updates

    # Policy kwargs matching the paper
    policy_kwargs = dict(
        hidden_dims=[256, 256, 256],
        dropout_rate=0.01,
        tau=0.005,
        gamma=0.99,
    )

    # Learning rate schedule
    def lr_schedule(progress_remaining: float) -> float:
        return 3e-4

    if args.resume:
        print(f"Loading model from {args.checkpoint_path}")
        model = SAC.load(
            args.checkpoint_path,
            env=vec_env,
            device=device,
            policy_kwargs=policy_kwargs,
        )
    else:
        model = SAC(
            policy=DroQPolicy,
            env=vec_env,
            learning_rate=lr_schedule,
            batch_size=256,
            buffer_size=1_000_000,  # 1M
            learning_starts=5000,
            train_freq=1,
            gradient_steps=1,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            target_update_interval=1,
            tensorboard_log=log_dir,
            device=device,
            policy_kwargs=policy_kwargs,
            verbose=1
        )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, monitor_callback, droq_callback],
            log_interval=1,
            tb_log_name=f"DroQ_{timestamp}"
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

if __name__ == "__main__":
    main() 