from IPython.display import HTML
from base64 import b64encode
import numpy as np
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env
from robopianist.suite.tasks.piano_with_shadow_hands import PianoWithShadowHands
import numpy as np
from data_processing.add_fingering_to_midi import add_fingering_from_annotation_file
from ppo_base import PPOAgent

# Instead, load the midi file then add the fingering annotations to it as a sequence,
# then convert the sequence to a midi_file object

midi_sequence = add_fingering_from_annotation_file(
        "/Users/almondgod/Repositories/robopianist/midi_files_cut/Guren no Yumiya Cut 14s.mid",
        "/Users/almondgod/Repositories/robopianist/data_processing/Guren no Yumiya Cut 14s_fingering v3.txt"
    )

# Create task
task = PianoWithShadowHands(
    midi=midi_sequence,
    n_steps_lookahead=1,
    n_seconds_lookahead=None,
    trim_silence=True,
    wrong_press_termination=False,
    initial_buffer_time=0.0,
    disable_fingering_reward=False,
    disable_forearm_reward=False,
    disable_colorization=False,
    disable_hand_collisions=False,
)

env = composer_utils.Environment(
    task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
)

env = PianoSoundVideoWrapper(
    env,
    record_every=100,
    camera_id="piano/back",
    record_dir=".",
)

env = CanonicalSpecWrapper(env)

# Create random state
random_state = np.random.RandomState(seed=42)  # or any seed you want

# Initialize the environment
timestep = env.reset()

# Get action spec to know the shape/bounds of actions
action_spec = env.action_spec()
observation_spec = env.observation_spec()
print(f"action_spec: {action_spec}")
print(f"observation_spec: {observation_spec}")

# Calculate total state dimension from observation spec dictionary
state_dim = sum(np.prod(spec.shape) for spec in observation_spec.values())
action_dim = env.action_spec().shape[0]

print(f"State dimension: {state_dim}")  # Debug print
print(f"Action dimension: {action_dim}")  # Debug print

# Initialize the agent with correct dimensions
agent = PPOAgent(
    state_dim=state_dim,  # Total flattened dimension of all observation arrays
    action_dim=action_dim,  # Dimension of the action space (45 for shadow hands)
    lr=3e-4,
    gamma=0.99,
    epsilon=0.5
)

# Loop
num_episodes = 100

for episode in range(num_episodes):
    print(f"Episode {episode}")
    timestep = env.reset()
    
    while not timestep.last():
        # Flatten and concatenate all observation arrays
        state = np.concatenate([
            timestep.observation['goal'].flatten(),
            timestep.observation['fingering'].flatten(),
            timestep.observation['piano/state'].flatten(),
            timestep.observation['piano/sustain_state'].flatten(),
            timestep.observation['rh_shadow_hand/joints_pos'].flatten(),
            timestep.observation['lh_shadow_hand/joints_pos'].flatten()
        ])
        
        # Get action from agent
        action, log_prob = agent.select_action(state)
        
        # Step environment
        prev_state = state  # Store current state for training
        timestep = env.step(action)
        
        # Get reward
        reward = [timestep.reward]
        
        # Flatten and concatenate next state
        next_state = np.concatenate([
            timestep.observation['goal'].flatten(),
            timestep.observation['fingering'].flatten(),
            timestep.observation['piano/state'].flatten(),
            timestep.observation['piano/sustain_state'].flatten(),
            timestep.observation['rh_shadow_hand/joints_pos'].flatten(),
            timestep.observation['lh_shadow_hand/joints_pos'].flatten()
        ])
        
        # Update agent
        agent.update(
            states=prev_state,
            actions=action,
            rewards=reward,
            log_probs=log_prob,
            next_states=next_state
        )

    print(f"Episode {episode} finished with reward {reward}")

