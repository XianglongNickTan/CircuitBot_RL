# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base Task class."""

import collections
import os
import random
import string
import tempfile

import cv2
import numpy as np
from utils import utils

import pybullet as p

import sys

from tasks.cameras import DaBai
from circuitbot.jaco_sim.jaco import Jaco
from utils.pathplanning.pathanalyzer import PathAnalyzer


rootdir = os.path.dirname(sys.modules['__main__'].__file__)
rootdir += "/assets"


obj_cube = rootdir + "/cube_4.obj"
obj_cuboid1 = rootdir + "/cuboid_4_4_8.obj"
obj_cuboid2 = rootdir + "/cuboid_4_16.obj"
obj_cuboid3 = rootdir + "/cuboid_8_8_4.obj"
obj_curve = rootdir + "/curve.obj"
obj_cylinder = rootdir + "/cylinder_4_4.obj"
obj_triangular_prism = rootdir + "/triangular_prism_4_8.obj"


OBJECTS = {
	'cube': obj_cube,
	'cuboid1': obj_cuboid1,
	'cuboid2': obj_cuboid2,
	'cuboid3': obj_cuboid3,
	'curve': obj_curve,
	'cylinder': obj_cylinder,
	'triangular_prism': obj_triangular_prism
}


class Task():
	"""Base Task class."""

	def __init__(self, env = None):
		"""Constructor.

		Args:
		  continuous: Set to `True` if you want the continuous variant.
		"""
		self.mode = 'train'

		self.weight_map = None

		self.env = env

		self.obj_type = OBJECTS

		# Workspace bounds.
		# self.pix_size = 0.005
		self.pix_size = 0.0025

		self.bounds = np.array([[0.1, 0.9], [-0.3, 0.3], [0, 0.3]])


		self.assets_root = None

		self.objects = []
		self.electrodeID = []

		self.pick_threshold = 0.05  ## m
		self.grip_z_offset = 0.07

		self.camera = DaBai.CONFIG

		self.arm = Jaco()

		self.analyzer = PathAnalyzer()



	def compare_object_base(self, pick_pos):
		move_object = None
		max_z = 0

		for object in self.objects:
			base, orin = p.getBasePositionAndOrientation(object)
			object_z = base[2]

			if pick_pos[0] - self.pick_threshold <= base[0] <= pick_pos[0] + self.pick_threshold:
				if pick_pos[1] - self.pick_threshold <= base[1] <= pick_pos[1] + self.pick_threshold:
					if object_z > max_z:
						max_z = object_z
						move_object = object

			else:
				continue

		return move_object


	def init_weight_map(self):
		weight_map = np.zeros([80, 56])

		return weight_map


	def update_weight_map(self, depth_map):
		# depth_map = np.asarray(depth_map)

		x, y = depth_map.shape[0:2]


		depth_map *= 100    ### convert to cm
		depth_map -= 1      ### minus plate height

		#### resize the map to weight map ######
		weight_map = cv2.resize(depth_map, (int(y / 0.01 * self.pix_size), int(x / 0.01 * self.pix_size)))

		return weight_map



	def set_add_electrode(self, black_x_offset=0.125, black_y_offset=0.055, white_x_offset=0.125, white_y_offset=-0.055):

		robot_x = 0.125
		robot_y = 0.055

		robot_row = int((self.bounds[0, 1] - self.bounds[0, 0] - robot_x) * 100 - 0.5)
		robot_column = int((self.bounds[1, 1] - robot_y) * 100 - 0.5)

		### create robot electrode ###
		utils.create_obj(p.GEOM_BOX,
						 mass=-1,
						 halfExtents=[0.0075, 0.0075, 0.0001],
						 rgbaColor=[0, 0, 0, 1],
						 basePosition=[robot_x + self.bounds[0, 0], robot_y, 0.01],
						 baseOrientation=[0, 0, 0, 1],
		                 object_list=self.electrodeID
						 )

		utils.create_obj(p.GEOM_BOX,
	 					 mass=-1,
	 	 				 halfExtents=[robot_x/2, 0.005, 0.0001],
	 					 rgbaColor=[0, 0, 0, 1],
	 					 basePosition=[robot_x/2 + self.bounds[0, 0], robot_y, 0.01],
	 					 baseOrientation=[0, 0, 0, 1],
					     object_list=self.electrodeID
		                 )

		utils.create_obj(p.GEOM_BOX,
						 mass=-1,
						 halfExtents=[0.0075, 0.0075, 0.0001],
						 rgbaColor=[1, 1, 1, 1],
						 basePosition=[robot_x + self.bounds[0, 0], -robot_y, 0.01],
						 baseOrientation=[0, 0, 0, 1],
						 object_list=self.electrodeID
		                 )

		utils.create_obj(p.GEOM_BOX,
						 mass=-1,
						 halfExtents=[robot_x/2, 0.005, 0.0001],
						 rgbaColor=[1, 1, 1, 1],
						 basePosition=[robot_x/2 + self.bounds[0, 0], -robot_y, 0.01],
						 baseOrientation=[0, 0, 0, 1],
						 object_list=self.electrodeID
						 )


		### create power source electrode ###
		utils.create_obj(p.GEOM_BOX,
						 mass=-1,
						 halfExtents=[0.0075, 0.0075, 0.0001],
						 rgbaColor=[0, 0, 0, 1],
						 basePosition=[self.bounds[0, 1] - black_x_offset, black_y_offset, 0.01],
						 baseOrientation=[0, 0, 0, 1],
						 object_list=self.electrodeID
						 )


		utils.create_obj(p.GEOM_BOX,
						 mass=-1,
						 halfExtents=[black_x_offset/2, 0.005, 0.0001],
						 rgbaColor=[0, 0, 0, 1],
						 basePosition=[self.bounds[0, 1] - black_x_offset / 2, black_y_offset, 0.01],
						 baseOrientation=[0, 0, 0, 1],
						 object_list=self.electrodeID
						 )

		utils.create_obj(p.GEOM_BOX,
						 mass=-1,
						 halfExtents=[0.0075, 0.0075, 0.0001],
						 rgbaColor=[1, 1, 1, 1],
						 basePosition=[self.bounds[0, 1] - white_x_offset, white_y_offset, 0.01],
						 baseOrientation=[0, 0, 0, 1],
						 object_list=self.electrodeID
						 )


		utils.create_obj(p.GEOM_BOX,
						 mass=-1,
						 halfExtents=[white_x_offset/2, 0.005, 0.0001],
						 rgbaColor=[1, 1, 1, 1],
						 basePosition=[self.bounds[0, 1] - white_x_offset / 2, white_y_offset, 0.01],
						 baseOrientation=[0, 0, 0, 1],
						 object_list=self.electrodeID
						 )

		source_row = int(89 - (self.bounds[0, 1] - self.bounds[0, 0] - black_x_offset) * 100 - 0.5)
		source_column = int((self.bounds[1, 1] - black_y_offset) * 100 - 0.5)

		self.analyzer.set_map(self.init_weight_map())

		self.analyzer.set_pathplan(0, [robot_column, robot_row], [source_column, source_row])
		self.analyzer.set_pathplan(1, [55 - robot_column, robot_row], [55 - source_column, source_row])

		# print([robot_column, robot_row], [source_column, source_row])
		# print([55 - robot_column, robot_row], [55 - source_column, source_row])




	def get_true_image(self):
		"""Get RGB-D orthographic heightmaps and segmentation masks."""

		# Capture near-orthographic RGB-D images and segmentation masks.
		color, depth, segm = self.env.render_camera(self.camera[0])

		# Combine color with masks for faster processing.
		color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

		# Reconstruct real orthographic projection from point clouds.
		hmaps, cmaps = utils.reconstruct_heightmaps(
			[color], [depth], self.camera, self.bounds, self.pix_size)

		# Split color back into color and masks.
		cmap = np.uint8(cmaps)[0, Ellipsis, :3]
		hmap = np.float32(hmaps)[0, Ellipsis]
		mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
		return cmap, hmap, mask