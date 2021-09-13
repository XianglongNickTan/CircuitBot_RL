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

		self.max_steps = 3

		self.env = env

		self.area_center = (0, 0)

		self.area_list = []

	def add_obstacles(self):
		super().add_obstacles()

		obstacle_type = [self.obj_type['cuboid1'],
		                 self.obj_type['cuboid2'],
		                 self.obj_type['curve'],
		                 self.obj_type['triangular_prism']]


		pos_index = random.choice([-1, 1])

		# self.area_center= self.add_forbidden_area(
		# 	top_left=[0.35, -0.2 + pos_index * random.random() / 10],
		# 	bottom_right=[0.45, 0.2 + pos_index * random.random() / 10])
		#
		#
		# self.area_center= self.add_forbidden_area(
		# 	top_left=[0.65, -0.2 - pos_index * random.random() / 10],
		# 	bottom_right=[0.75, 0.2 - pos_index * random.random() / 10])
		base = 0.5 + 1.5 * random.choice([-1, 1]) * random.random() / 10

		self.area_center = self.add_forbidden_area(
			top_left=[base, -0.2 + pos_index * random.random() / 10],
			bottom_right=[base + 0.1, 0.2 + pos_index * random.random() / 10])



		pos_index = random.choice([-1, 1])


		# utils.create_obj(p.GEOM_MESH,
		# 							mass=0.01,
		# 							use_file=self.obj_type['cuboid2'],
		# 							rgbaColor=[1,0,1,1],
		# 							basePosition=[0.5 + 0.2*pos_index + (2 * random.random() - 1) / 10,
		# 							              0.10 * (2 * random.random() - 1), 0.03],
		# 							baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
		#                             object_list=self.objects
		# 							)
		#
		#
		# utils.create_obj(p.GEOM_MESH,
		# 							mass=0.01,
		# 							use_file=self.obj_type['curve'],
		# 							rgbaColor=[0,1,1,1],
		# 							basePosition=[0.5 - 0.2*pos_index - (2 * random.random() - 1) / 10,
		# 							              0.1 * (2 * random.random() - 1), 0.03],
		# 							baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/4]),
		#                             object_list=self.objects
		# 							)

		base_1 = [0.5 + 0.2*pos_index + (2 * random.random() - 1) / 10, 0.1 * (2 * random.random() - 1), 0.03]

		utils.create_obj(p.GEOM_MESH,
									mass=0.01,
									use_file=self.random_object(),
									rgbaColor=self.random_color(),
									basePosition=base_1,
									baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
		                            object_list=self.objects
									)


		base_2 = [0.5 - 0.2*pos_index - (2 * random.random() - 1) / 10, 0.1 * (2 * random.random() - 1), 0.03]

		utils.create_obj(p.GEOM_MESH,
									mass=0.01,
									use_file=self.random_object(),
									rgbaColor=self.random_color(),
									basePosition=base_2,
									baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
		                            object_list=self.objects
									)


		pos_index = random.choice([-1, 1])

		base_3 = [0.5 + pos_index*random.random() / 10, 0.15 * (2 * random.random() - 1), 0.03]

		distance_ok = self.compare_object_distance(base_3, base_1, base_2)

		while not distance_ok:
			base_3 = [0.5 + pos_index * random.random() / 10, 0.15 * (2 * random.random() - 1), 0.03]
			distance_ok = self.compare_object_distance(base_3, base_1, base_2)

		utils.create_obj(p.GEOM_MESH,
									mass=0.01,
									use_file=self.obj_type['bridge2'],
									rgbaColor=self.random_color(),
									basePosition=base_3,
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


			place_z = self.grip_z_offset

			if self.grap_num == 2:
				place_pos = (self.area_center[0], 0, place_z)



			else:
				place_pos = (base[0], 0.26, place_z)


			place_orin = p.getQuaternionFromEuler([0, -np.pi, 0])

			place_pose = (np.asarray(place_pos), np.asarray(place_orin))

			self.grap_num += 1


			return {'pose0': pick_pose, 'pose1': place_pose}

		return OracleAgent(act)


