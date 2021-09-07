import collections
import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

import pybullet as p

class PickAndPlaceTask(Task):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_steps = 3

  def reset(self, env):
    super().reset(env)
    block_id = self.add_block(env)
    targ_pose = self.add_fixture(env)
    # self.goals.append(
    #     ([block_id], [2 * np.pi], [[0]], [targ_pose], 'pose', None, 1.))
    self.goals.append(([(block_id, (2 * np.pi, None))], np.int32([[1]]),
                       [targ_pose], False, True, 'pose', None, 1))

  def add_block(self, env):
    """Add L-shaped block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'insertion/ell.urdf'
    pose = self.get_random_pose(env, size)
    return env.add_object(urdf, pose)

  def add_fixture(self, env):
    """Add L-shaped fixture to place block."""
    size = (0.1, 0.1, 0.04)
    urdf = 'insertion/fixture.urdf'
    pose = self.get_random_pose(env, size)
    env.add_object(urdf, pose, 'fixed')
    return pose
  
  def get_discrete_oracle_agent(self, env):
    OracleAgent = collections.namedtuple('OracleAgent', ['act'])

    def act(obs, info):  # pylint: disable=unused-argument
      """Calculate action."""

      # Oracle uses perfect RGB-D orthographic images and segmentation masks.
      _, hmap, obj_mask = self.get_true_image(env)

      # Unpack next goal step.
      objs, matches, targs, replace, rotations, _, _, _ = self.goals[0]

      # Match objects to targets without replacement.
      if not replace:

        # Modify a copy of the match matrix.
        matches = matches.copy()

        # Ignore already matched objects.
        for i in range(len(objs)):
          object_id, (symmetry, _) = objs[i]
          pose = p.getBasePositionAndOrientation(object_id)
          targets_i = np.argwhere(matches[i, :]).reshape(-1)
          for j in targets_i:
            if self.is_match(pose, targs[j], symmetry):
              matches[i, :] = 0
              matches[:, j] = 0

      # Get objects to be picked (prioritize farthest from nearest neighbor).
      nn_dists = []
      nn_targets = []
      for i in range(len(objs)):
        object_id, (symmetry, _) = objs[i]
        xyz, _ = p.getBasePositionAndOrientation(object_id)
        targets_i = np.argwhere(matches[i, :]).reshape(-1)
        if len(targets_i) > 0:  # pylint: disable=g-explicit-length-test
          targets_xyz = np.float32([targs[j][0] for j in targets_i])
          dists = np.linalg.norm(
              targets_xyz - np.float32(xyz).reshape(1, 3), axis=1)
          nn = np.argmin(dists)
          nn_dists.append(dists[nn])
          nn_targets.append(targets_i[nn])

        # Handle ignored objects.
        else:
          nn_dists.append(0)
          nn_targets.append(-1)
      order = np.argsort(nn_dists)[::-1]

      # Filter out matched objects.
      order = [i for i in order if nn_dists[i] > 0]

      pick_mask = None
      for pick_i in order:
        pick_mask = np.uint8(obj_mask == objs[pick_i][0])

        # Erode to avoid picking on edges.
        # pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))

        if np.sum(pick_mask) > 0:
          break

      # Trigger task reset if no object is visible.
      if pick_mask is None or np.sum(pick_mask) == 0:
        self.goals = []
        print('Object for pick is not visible. Skipping demonstration.')
        return

      # Get picking pose.
      pick_prob = np.float32(pick_mask)
      pick_pix = utils.sample_distribution(pick_prob)
      # For "deterministic" demonstrations on insertion-easy, use this:
      # pick_pix = (160,80)
      pick_pos = utils.pix_to_xyz(pick_pix, hmap,
                                  self.bounds, self.pix_size)
      pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

      # Get placing pose.
      targ_pose = targs[nn_targets[pick_i]]  # pylint: disable=undefined-loop-variable
      obj_pose = p.getBasePositionAndOrientation(objs[pick_i][0])  # pylint: disable=undefined-loop-variable
      if not self.sixdof:
        obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1])
        obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
        obj_pose = (obj_pose[0], obj_quat)
      world_to_pick = utils.invert(pick_pose)
      obj_to_pick = utils.multiply(world_to_pick, obj_pose)
      pick_to_obj = utils.invert(obj_to_pick)
      place_pose = utils.multiply(targ_pose, pick_to_obj)

      # Rotate end effector?
      if not rotations:
        place_pose = (place_pose[0], (0, 0, 0, 1))

      place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))

      return {'pose0': pick_pose, 'pose1': place_pose}

    return OracleAgent(act)

