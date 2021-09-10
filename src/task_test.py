import os, sys

from absl import app
from absl import flags

import numpy as np

from ravens.dataset import Dataset

from env.environment import Environment
# from tasks.clear_one_obstacle import ClearOneObstacle
from tasks.all_in_one import AllInOne
import tasks



flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', True, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 100, '')
flags.DEFINE_integer('steps_per_seg', 3, '')
flags.DEFINE_string('task', 'all-in-one', '')

FLAGS = flags.FLAGS


def main(unused_argv):

  # Initialize environment and task.
  env = Environment(
      disp=FLAGS.disp,
      shared_memory=FLAGS.shared_memory,
      hz=240)
  task = tasks.names[FLAGS.task](env)
  task.mode = FLAGS.mode

  # Initialize scripted oracle agent and dataset.
  agent = task.get_discrete_oracle_agent()
  dataset = Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{task.mode}'))

  # Train seeds are even and test seeds are odd.
  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if (task.mode == 'test') else -2

  max_steps = task.max_steps

  # Collect training data from oracle demonstrations.
  while dataset.n_episodes < FLAGS.n:
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}')
    episode = []
    seed += 2
    np.random.seed(seed)
    env.set_task(task)
    obs = env.reset()
    info = None
    reward = 0
    for _ in range(max_steps):
      act = agent.act(obs, info)
      episode.append((obs, act, reward, info))
      obs, reward, done, info = env.step(act)
      print(f'This Reward: {reward} Done: {done}')
      if done:
        break
    episode.append((obs, None, reward, info))

if __name__ == '__main__':
  app.run(main)
