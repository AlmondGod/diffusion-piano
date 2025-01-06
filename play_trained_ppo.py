from ppo_base import PPOAgent
import torch
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file
import numpy as np
from robopianist.wrappers import PianoSoundVideoWrapper
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
from mujoco_utils import composer_utils

class VectorizedRecordingPianoEnv:
    def __init__(self, num_envs, midi_sequence):
        self.num_envs = num_envs
        self.envs = []
        
        for _ in range(num_envs):
            # Create task
            task = PianoWithShadowHands(
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
            env = composer_utils.Environment(
                task=task, 
                strip_singleton_obs_buffer_dim=True, 
                recompile_physics=True
            )

            env = PianoSoundVideoWrapper(
                env,
                record_every=1,
                camera_id="piano/back",
                record_dir=".",
            )

            env = CanonicalSpecWrapper(env)
            
            self.envs.append(env)
            
    def reset(self):
        """Reset all environments"""
        observations = [env.reset().observation for env in self.envs]
        return self._stack_obs(observations)
    
    def step(self, actions):
        """Step all environments in parallel"""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        next_obs = [timestep.observation for timestep in results]
        rewards = [timestep.reward for timestep in results]
        dones = [timestep.last() for timestep in results]
        
        return self._stack_obs(next_obs), np.array(rewards), np.array(dones)
    
    def _stack_obs(self, observations):
        """Stack observations from all environments"""
        stacked_obs = {}
        for key in observations[0].keys():
            stacked_obs[key] = np.stack([obs[key] for obs in observations])
        return stacked_obs

midi_sequence = add_fingering_from_annotation_file(
        "/Users/almondgod/Repositories/robopianist/midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "/Users/almondgod/Repositories/robopianist/data_processing/Guren no Yumiya Cut 14s_fingering v3.txt"
    )

# create envrionment
num_envs = 1  
vec_env = VectorizedRecordingPianoEnv(num_envs, midi_sequence)

# Get dimensions from first environment
observation_spec = vec_env.envs[0].observation_spec()
action_spec = vec_env.envs[0].action_spec()

state_dim = sum(np.prod(spec.shape) for spec in observation_spec.values())
action_dim = action_spec.shape[0]

print(f"State dimension: {state_dim}")  # Debug print
print(f"Action dimension: {action_dim}")  # Debug print


agent = PPOAgent(state_dim, action_dim)

# Load from checkpoint to resume training
# episode, history = agent.load_checkpoint('checkpoints/checkpoint_episode_50.pt')

# Or load just the model weights for inference
model_weights = torch.load('/Users/almondgod/Repositories/robopianist/models/final_model.pt')
agent.actor.load_state_dict(model_weights['actor_state_dict'])
agent.critic.load_state_dict(model_weights['critic_state_dict'])

# Play the trained model

# Loop
num_episodes = 1

# Training loop
for episode in range(num_episodes):
    obs = vec_env.reset()
    episode_rewards = np.zeros(num_envs)
    
    while True:
        # Stack and flatten observations from all environments
        states = np.stack([
            np.concatenate([
                obs['goal'][i].flatten(),
                obs['fingering'][i].flatten(),
                obs['piano/state'][i].flatten(),
                obs['piano/sustain_state'][i].flatten(),
                obs['rh_shadow_hand/joints_pos'][i].flatten(),
                obs['lh_shadow_hand/joints_pos'][i].flatten()
            ]) for i in range(num_envs)
        ])
        
        # Get actions for all environments
        actions, log_probs = agent.select_actions(states)
        
        # Step all environments
        next_obs, rewards, dones = vec_env.step(actions)
        
        # Flatten next_states the same way as states
        next_states = np.stack([
            np.concatenate([
                next_obs['goal'][i].flatten(),
                next_obs['fingering'][i].flatten(),
                next_obs['piano/state'][i].flatten(),
                next_obs['piano/sustain_state'][i].flatten(),
                next_obs['rh_shadow_hand/joints_pos'][i].flatten(),
                next_obs['lh_shadow_hand/joints_pos'][i].flatten()
            ]) for i in range(num_envs)
        ])
        
        episode_rewards += rewards
        
        
        if all(dones):
            break
            
        obs = next_obs