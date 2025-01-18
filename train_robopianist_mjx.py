import os
import time
from pathlib import Path
import jax
from brax.training.agents.ppo import train as ppo
import wandb
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file
from robopianist_mjx import RoboPianistMJX
import pickle

# Configure environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"] = "4294967296"  # 4GB

def limit_jax_mem(limit):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"

def main():
    # Setup
    limit_jax_mem(0.6)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f"checkpoints/robopianist_mjx_{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load MIDI sequence
    midi_sequence = add_fingering_from_annotation_file(
        "./midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "./data_processing/Guren no Yumiya Cut 14s_fingering v3.txt"
    )

    # Create environment
    num_envs = 1024  # Can be much larger with GPU acceleration
    env = RoboPianistMJX(midi_sequence, num_envs=num_envs)

    # Initialize wandb
    wandb.init(
        project="robopianist-mjx",
        config={
            "num_envs": num_envs,
            "num_timesteps": 5_000_000,
            "episode_length": 1000,
            "batch_size": 1024,
            "learning_rate": 3e-4,
        }
    )

    # Training configuration
    train_fn = ppo.train(
        environment=env,
        num_timesteps=5_000_000,
        episode_length=1000,
        num_envs=num_envs,
        batch_size=1024,
        num_minibatches=32,
        num_updates=1,
        reward_scaling=0.1,
        entropy_cost=1e-2,
        normalize_observations=True,
        action_repeat=1,
        progress_fn=wandb.log
    )

    try:
        # Train
        make_inference_fn, params, metrics = train_fn(environment=env)
        
        # Save final model
        checkpoint = {
            'params': params,
            'metrics': metrics,
            'config': {
                'num_envs': num_envs,
                'obs_dim': env._obs_dim,
                'act_dim': env._act_dim
            }
        }
        final_path = checkpoint_dir / 'final_model.pkl'
        with open(final_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Model saved to {final_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted!")
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 