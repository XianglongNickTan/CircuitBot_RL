import os, sys

from absl import app
from absl import flags

import numpy as np

from ravens.dataset import Dataset

from map_env.environment import Environment
from training.tasks.clear_obstacles import ClearObstaclesTask



flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 1000, '')
flags.DEFINE_integer('steps_per_seg', 3, '')

FLAGS = flags.FLAGS


env = Environment()

task = ClearObstaclesTask(env)
task.mode = FLAGS.mode

# Initialize scripted oracle agent and dataset.
agent = task.get_discrete_oracle_agent()
dataset = Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{task.mode}'))

# Train seeds are even and test seeds are odd.
seed = dataset.max_seed
if seed < 0:
	seed = -1 if (task.mode == 'test') else -2

max_steps = task.max_steps
