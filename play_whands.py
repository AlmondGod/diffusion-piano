# @title All imports required for this tutorial
from IPython.display import HTML
from base64 import b64encode
import numpy as np
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env

# @title Helper functions
# Reference: https://stackoverflow.com/a/60986234.
def play_video(filename: str):
    mp4 = open(filename, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

    return HTML(
        """
  <video controls>
        <source src="%s" type="video/mp4">
  </video>
  """
        % data_url
    )

task = piano_with_shadow_hands.PianoWithShadowHands(
    change_color_on_activation=True,
    midi=music.load("/Users/almondgod/Repositories/robopianist/Attack on Titan OP1 - Guren no Yumiya.mid.mid"),
    trim_silence=True,
    control_timestep=0.05,
    gravity_compensation=True,
    primitive_fingertip_collisions=False,
    reduced_action_space=False,
    n_steps_lookahead=10,
    disable_fingering_reward=False,
    disable_forearm_reward=False,
    disable_colorization=False,
    disable_hand_collisions=False,
    attachment_yaw=0.0,
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

action_spec = env.action_spec()
print(f"Action dimension: {action_spec.shape}")

timestep = env.reset()
dim = 0
for k, v in timestep.observation.items():
    print(f"\t{k}: {v.shape} {v.dtype}")
    dim += int(np.prod(v.shape))
print(f"Observation dimension: {dim}")

class Policy:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._idx = 0
        self._actions = np.load("twinkle_twinkle_actions.npy")
        # Store the length of actions for bounds checking
        self._max_steps = len(self._actions)

    def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
        del timestep  # Unused.
        # Check if we've reached the end of the actions
        if self._idx >= self._max_steps:
            return np.zeros_like(self._actions[0])  # Return zero action at the end
        actions = self._actions[self._idx]
        self._idx += 1
        return actions
    
policy = Policy()

timestep = env.reset()
while not timestep.last():
    action = policy(timestep)
    timestep = env.step(action)

play_video(env.latest_filename)