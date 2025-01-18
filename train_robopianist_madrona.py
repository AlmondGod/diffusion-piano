import os
from pathlib import Path
import time
import jax
from brax.training.agents.ppo import train as ppo
import wandb
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file
from robopianist_madrona import RoboPianistMadrona, RoboPianistConfig
import pickle
import dataclasses

# Configure environment variables for Madrona caching
os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "build/kernel_cache"
os.environ["MADRONA_BVH_KERNEL_CACHE"] = "build/bvh_cache"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def main():
    # Setup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f"checkpoints/robopianist_madrona_{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load MIDI sequence
    midi_sequence = add_fingering_from_annotation_file(
        "./midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "./data_processing/Guren no Yumiya Cut 14s_fingering v3.txt"
    )

    # Create environment
    config = RoboPianistConfig(
        num_worlds=1024,
        render_width=64,
        render_height=64
    )
    env = RoboPianistMadrona(midi_sequence, config)

    # Initialize wandb
    wandb.init(
        project="robopianist-madrona",
        config={
            "num_worlds": config.num_worlds,
            "num_timesteps": 5_000_000,
            "episode_length": config.episode_length,
            "render_size": (config.render_width, config.render_height),
        }
    )

    try:
        # Train using Brax's PPO
        train_fn = ppo.train(
            environment=env,
            num_timesteps=5_000_000,
            episode_length=config.episode_length,
            num_envs=config.num_worlds,
            batch_size=1024,
            progress_fn=wandb.log
        )
        
        make_inference_fn, params, metrics = train_fn(environment=env)
        
        # Save model
        checkpoint = {
            'params': params,
            'metrics': metrics,
            'config': dataclasses.asdict(config)
        }
        with open(checkpoint_dir / 'final_model.pkl', 'wb') as f:
            pickle.dump(checkpoint, f)
            
    except KeyboardInterrupt:
        print("Training interrupted!")
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 