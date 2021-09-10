import collections
import random

import math
import numpy as np
from tasks.task import Task
# from ravens.utils import utils
from utils import utils
from gym import spaces
import cv2

import os, sys
import pybullet as p

from tasks.task import Task


class AllInOne(Task):
	""" remove one cube in the path"""

	def __init__(self,
				 env):

		super().__init__()

		self.max_steps = 2

		self.env = env

		self.area_center = (0, 0)

		self.area_list = []

	def add_obstacles(self):
		super().add_obstacles()




		pos_index = random.randint(0,1)

		base_x = 0.3 + random.random()/10

		utils.create_obj(p.GEOM_MESH,
									mass=0.01,
									use_file=self.obj_type['cuboid2'],
									rgbaColor=self.set_color(),
									basePosition=[base_x + 0.25*pos_index,
									              0.10 * (2 * random.random() - 1), 0.03],
									baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
		                            object_list=self.objects
									)

		self.area_center= self.add_forbidden_area(
			top_left=[0.8 - base_x, -0.2 + random.random() / 10],
			bottom_right=[0.8 - base_x + 0.07, 0.2 + random.random() / 10])


		base_x = 0.55 + random.random() / 10

		utils.create_obj(p.GEOM_MESH,
									mass=0.01,
									use_file=self.obj_type['bridge'],
									rgbaColor=self.set_color(),
									basePosition=[base_x - 0.25 *pos_index,
									              0.10 * (2 * random.random() - 1), 0.03],
									baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
		                            object_list=self.objects
									)







	def apply_action(self, action=None):
		pick_pos = action['pose0']
		place_pos = action['pose1']

		# move_object = self.objects[0]
		move_object = self.compare_object_base(pick_pos[0], self.objects)

		pick_pos[0][2] += self.grip_z_offset
		place_pos[0][2] += self.grip_z_offset

		self.arm.pick_place_object(move_object, pick_pos[0], pick_pos[1], place_pos[0], place_pos[1])


	def remove_objects(self):
		for object in self.objects:
			p.removeBody(object)
		self.objects = []

		for object in self.electrodeID:
			p.removeBody(object)
		self.electrodeID = []

		for object in self.forbidden_area:
			p.removeBody(object)
		self.forbidden_area = []

		for object in self.ink_path:
			p.removeBody(object)
		self.ink_path = []
	#

	def reward(self):
		weight_map = self.get_weight_map()
		reward = self._get_reward(weight_map, self.area_list)

		return reward, None



	def get_discrete_oracle_agent(self):
		OracleAgent = collections.namedtuple('OracleAgent', ['act'])

		def act(obs, info):  # pylint: disable=unused-argument
			"""Calculate action."""
			# self._update_weight_map()

			move_object = self.objects[self.grap_num]

			base, pick_orin = p.getBasePositionAndOrientation(move_object)

			base = np.asarray(base)

			base[2] += self.grip_z_offset

			pick_pos = base

			pick_orin = p.getQuaternionFromEuler([0, -np.pi, p.getEulerFromQuaternion(pick_orin)[2]])

			pick_pose = (np.asarray(pick_pos), np.asarray(pick_orin))


			place_z = 0.04 + self.grip_z_offset

			if self.grap_num == 0:
				place_pos = (base[0], 0.22, place_z)


			else:
				place_pos = (self.area_center[0], self.area_center[1], place_z)


			place_orin = p.getQuaternionFromEuler([0, -np.pi, 0])

			place_pose = (np.asarray(place_pos), np.asarray(place_orin))

			self.grap_num += 1

			return {'pose0': pick_pose, 'pose1': place_pose}

		return OracleAgent(act)


