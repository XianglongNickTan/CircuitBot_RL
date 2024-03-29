import collections
import random

import numpy as np
from utils import utils


import pybullet as p

from tasks.task import Task


class ConstructOneBridge(Task):
	""" remove one cube in the path"""

	def __init__(self,
				 env):

		super().__init__()

		self.max_steps = 1

		self.env = env

		self.grap_num = 0
		self.area_center = (0, 0)

		self.area_list = []

		self.remove_zone()

	def add_obstacles(self):

		obstacle_type = self.obj_type['bridge1']


		base_x = 0.5 + 2 * (2 * random.random() - 1) / 10

		utils.create_obj(p.GEOM_MESH,
									mass=0.01,
									use_file=obstacle_type,
									rgbaColor=self.random_color(),
									basePosition=[base_x,
									              0.10 * (2 * random.random() - 1), 0.03],
									baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
		                            object_list=self.objects
									)

		# base_x = 0.5 + 2 * (2 * random.random() - 1) / 10
		#
		# self.area_center = self.add_forbidden_area(
		#                                 top_left=[base_x, -0.2 + random.random() / 10],
	    #                                 bottom_right=[base_x + 0.07, 0.2 + random.random() / 10])

		pos_index = random.choice([-1, 1])

		base = 0.5 + 1.5 * random.choice([-1, 1]) * random.random() / 10

		self.area_center = self.add_forbidden_area(
			top_left=[base, -0.2 + pos_index * random.random() / 10],
			bottom_right=[base + 0.03 + random.random() / 10, 0.2 + pos_index * random.random() / 10])


	def apply_action(self, action=None):
		pick_pos = action['pose0']
		place_pos = action['pose1']

		# move_object = self.objects[0]
		move_object = self.compare_object_base(pick_pos[0], self.objects)

		pick_pos[0][2] += self.grip_z_offset
		place_pos[0][2] += self.grip_z_offset

		self.arm.pick_place_object(move_object, pick_pos[0], pick_pos[1], place_pos[0], place_pos[1])


	def remove_zone(self):
		for object in self.waste_zone:
			p.removeBody(object)




	def get_discrete_oracle_agent(self):
		OracleAgent = collections.namedtuple('OracleAgent', ['act'])

		def act(obs, info):  # pylint: disable=unused-argument
			"""Calculate action."""
			# self._update_weight_map()

			move_object = self.objects[0]

			base, pick_orin = p.getBasePositionAndOrientation(move_object)

			base = np.asarray(base)

			base[2] += self.grip_z_offset

			pick_pos = base

			pick_orin = p.getQuaternionFromEuler([0, -np.pi, p.getEulerFromQuaternion(pick_orin)[2]])

			pick_pose = (np.asarray(pick_pos), np.asarray(pick_orin))


			place_z = 0.04 + self.grip_z_offset

			place_pos = (self.area_center[0], self.area_center[1], place_z)

			place_orin = p.getQuaternionFromEuler([0, -np.pi, 0])

			place_pose = (np.asarray(place_pos), np.asarray(place_orin))

			self.grap_num += 1

			return {'pose0': pick_pose, 'pose1': place_pose}

		return OracleAgent(act)


