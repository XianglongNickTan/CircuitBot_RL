import collections
import random

import numpy as np
from utils import utils

import pybullet as p

from tasks.task import Task


class ClearObstacles(Task):
	""" remove one cube in the path"""

	def __init__(self,
				 env):

		super().__init__()

		self.max_steps = 3

		self.env = env

		self.grap_num = 0


	def add_obstacles(self):
		obstacle_type = [self.obj_type['cuboid1'],
		                 self.obj_type['cuboid2'],
		                 self.obj_type['triangular_prism']]

		obstacle = random.choice(obstacle_type)

		utils.create_obj(p.GEOM_MESH,
									mass=0.01,
									use_file=obstacle,
									rgbaColor=self.random_color(),
									basePosition=[0.3 + random.random()/20,
									              0.10 * (2 * random.random() - 1), 0.03],
									baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
		                            object_list=self.objects
									)

		obstacle = random.choice(obstacle_type)

		utils.create_obj(p.GEOM_MESH,
									mass=0.01,
									use_file=obstacle,
									rgbaColor=self.random_color(),
									basePosition=[0.5 + random.random()/20,
									              0.10 * (2 * random.random() - 1), 0.03],
									baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
		                            object_list=self.objects
									)

		obstacle = random.choice(obstacle_type)

		utils.create_obj(p.GEOM_MESH,
									mass=0.01,
									use_file=obstacle,
									rgbaColor=self.random_color(),
									basePosition=[0.7 + random.random()/20,
									              0.10 * (2 * random.random() - 1), 0.03],
									baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
		                            object_list=self.objects
									)

		self.pick_list = self.objects.copy()


	def apply_action(self, action=None):
		pick_pos = action['pose0']
		place_pos = action['pose1']

		move_object = self.compare_object_base(pick_pos[0], self.objects)

		pick_pos[0][2] += self.grip_z_offset
		place_pos[0][2] += self.grip_z_offset

		self.arm.pick_place_object(move_object, pick_pos[0], pick_pos[1], place_pos[0], place_pos[1])




	def get_discrete_oracle_agent(self):
		OracleAgent = collections.namedtuple('OracleAgent', ['act'])

		def act(obs, info):  # pylint: disable=unused-argument
			"""Calculate action."""
			# self._update_weight_map()


			move_object = random.choice(self.pick_list)
			self.pick_list.remove(move_object)

			base, pick_orin = p.getBasePositionAndOrientation(move_object)

			base = np.asarray(base)

			base[2] += self.grip_z_offset

			pick_pos = base

			pick_orin = p.getQuaternionFromEuler([0, -np.pi, p.getEulerFromQuaternion(pick_orin)[2]])

			pick_pose = (np.asarray(pick_pos), np.asarray(pick_orin))


			place_z = self.grip_z_offset

			place_y = -0.27 + 0.07 * self.grap_num

			place_pos = (0.22, place_y, place_z)

			place_orin = p.getQuaternionFromEuler([0, -np.pi, 0])

			place_pose = (np.asarray(place_pos), np.asarray(place_orin))

			self.grap_num += 1

			return {'pose0': pick_pose, 'pose1': place_pose}

		return OracleAgent(act)


