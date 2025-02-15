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
#I instead have to load the midi file then add the fingering annotations to it as a sequence, then convert the sequence
# to a midi_file object

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
    record_every=1,
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

print(f"action_spec: {action_spec}")

# Loop
num_episodes = 10

for episode in range(num_episodes):
    print(f"Episode {episode}")
    timestep = env.reset()
    
    while not timestep.last():
        
        # Generate actions here 
        action = np.random.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum,
            size=action_spec.shape
        )
        
        # Step the environment
        timestep = env.step(action)
        
        # Get reward
        reward = timestep.reward