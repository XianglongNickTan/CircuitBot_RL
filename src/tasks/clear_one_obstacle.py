import collections
import random

import numpy as np
from tasks.task import Task
# from ravens.utils import utils
from utils import utils
from gym import spaces
import cv2

import os, sys
import pybullet as p

from tasks.task import Task


class ClearObstaclesTask(Task):
	""" remove one cube in the path"""

	def __init__(self,
				 env):

		super().__init__()

		self.max_steps = 1

		self.env = env

		self.grab_num = 0

		# self.action_space = spaces.Box(
		# 	low=np.array([22 * self.pixel_ratio, 8 * self.pixel_ratio, 22 * self.pixel_ratio, 8 * self.pixel_ratio, 0]),
		# 	high=np.array([58 * self.pixel_ratio, 47 * self.pixel_ratio, 58 * self.pixel_ratio, 47 * self.pixel_ratio, 1]),
		# 	dtype=np.int)


	def add_obstacles(self):
		obstacle_type = [self.obj_type['cube'],
		                 self.obj_type['cuboid1'],
		                 self.obj_type['cuboid2'],
		                 self.obj_type['cuboid3']]

		obstacle = obstacle_type[random.randint(0, 3)]

		utils.create_obj(p.GEOM_MESH,
									mass=0.01,
									use_file=obstacle,
									rgbaColor=utils.COLORS['red'],
									basePosition=[0.3 + 4 * random.random() / 10,
									              0.10 * (2 * random.random() - 1), 0.03],
									baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
		                            object_list=self.objects
									)
		#



	def apply_action(self, action=None):
		pick_pos = action['pose0']
		place_pos = action['pose1']

		move_object = self.objects[0]

		if pick_pos[0][2] < self.grip_z_offset:
			pick_pos[0][2] += self.grip_z_offset

		if place_pos[0][2] < self.grip_z_offset:
			place_pos[0][2] += self.grip_z_offset

		self.arm.pick_place_object(move_object, pick_pos[0], pick_pos[1], place_pos[0], place_pos[1])

		self.grab_num += 1

		# pass

	def remove_objects(self):
		for object in self.objects:
			p.removeBody(object)
		self.objects = []

		for object in self.electrodeID:
			p.removeBody(object)
		self.electrodeID = []


	def reset(self):
		self.remove_objects()
		self.grab_num = 0

		self.set_add_electrode()
		self.add_obstacles()





	def reward(self, depth_map):
		reward = 0

		weight_map = self.update_weight_map(depth_map)

		self.analyzer.set_map(weight_map)
		self.analyzer.search()

		success_1, path_1, cost_1 = self.analyzer.get_result(0)
		success_2, path_2, cost_2 = self.analyzer.get_result(1)

		if success_1:
			reward += 100
			reward -= cost_1

		if success_2:
			reward += 100
			reward -= cost_2

		# self.analyzer.draw_map_3D()

		print(reward)
		return reward


	def done(self):
		return None

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
			if base[1] > 0:
				place_y = 0.22

			else:
				place_y = -0.22
			place_pos = (base[0], place_y, place_z)

			place_orin = p.getQuaternionFromEuler([0, -np.pi, 0])

			place_pose = (np.asarray(place_pos), np.asarray(place_orin))


			return {'pose0': pick_pose, 'pose1': place_pose}

		return OracleAgent(act)


