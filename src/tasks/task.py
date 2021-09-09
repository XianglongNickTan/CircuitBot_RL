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
import sys
import math

import cv2
import numpy as np
from tasks import cameras
import pybullet as p
from utils import utils
from circuitbot.jaco_sim.jaco import Jaco
from utils.pathplanning.pathanalyzer import PathAnalyzer


rootdir = os.path.dirname(sys.modules['__main__'].__file__)
rootdir += "/assets/blocks"


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


COLORS = {
	'blue': [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0, 1],
	'red': [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0, 1],
	'green': [089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0, 1],
	'orange': [242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0, 1],
	'yellow': [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0, 1],
	'purple': [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0, 1],
	'pink': [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0, 1],
	'cyan': [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0, 1],
	'brown': [156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0, 1],
	'gray': [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0, 1]
}




class Task:
	"""Base Task class."""

	def __init__(self, env=None):
		"""Constructor.

		Args:
		  continuous: Set to `True` if you want the continuous variant.
		"""
		self.mode = 'train'
		self.weight_map = None
		self.env = env

		self.obj_type = OBJECTS
		self.color = COLORS
		self.oracle_cams = cameras.Oracle.CONFIG

		# Workspace bounds.
		# self.pix_size = 0.005
		self.pix_size = 0.005

		self.bounds = np.array([[0.1, 0.9], [-0.3, 0.3], [0, 0.3]])


		self.assets_root = None

		self.objects = []
		self.electrodeID = []
		self.nono_area = []

		self.pick_threshold = 0.03  ## m
		self.grip_z_offset = 0.07


		self.arm = Jaco()

		self.analyzer = PathAnalyzer()

		self.goals = []
		self.progress = 0
		self._rewards = 0
		self.color_list = []

		self.get_color()

		self.obj_color = None



	def _get_reward(self, weight_map):
		self.analyzer.set_map(weight_map)
		self.analyzer.search()

		success_1, path_1, cost_1 = self.analyzer.get_result(0)
		success_2, path_2, cost_2 = self.analyzer.get_result(1)

		min_cost_1 = self.euler_dist(0)
		min_cost_2 = self.euler_dist(1)

		reward_1 = 5 ** ((min_cost_1 - cost_1) / min_cost_1) if success_1 else 0
		reward_2 = 5 ** ((min_cost_2 - cost_2) / min_cost_2) if success_2 else 0

		# self.analyzer.draw_map_3D()
		# self.analyzer.draw_map_3D_only()

		# print((reward_1 + reward_2) / 2)
		return (reward_1 + reward_2) / 2


	def euler_dist(self, no):
		x = self.analyzer.path_planners[no].departure.x - self.analyzer.path_planners[no].destination.x
		y = self.analyzer.path_planners[no].departure.y - self.analyzer.path_planners[no].destination.y
		z = self.analyzer.path_planners[no].departure.z - self.analyzer.path_planners[no].destination.z
		return math.sqrt(x ** 2 + y ** 2 + z ** 2) * 0.975


	def get_color(self):
		color = self.color.values()
		for i in color:
			self.color_list.append(i)


	def set_color(self):
		color_num = random.randint(0, 9)
		return self.color_list[color_num]


	def compare_object_base(self, pick_pos, object_list):
		move_object = None
		max_z = 0

		for object in object_list:
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
		weight_map = np.zeros([80, 60])

		return weight_map


	def get_weight_map(self):

		_, depth_map, _ = self.get_true_image(self.env)
		x, y = depth_map.shape[0:2]

		depth_map *= 100    ### convert to cm
		depth_map -= 1      ### minus plate height

		#### resize the map to weight map ######
		depth_map = cv2.resize(depth_map, (int(y / 0.01 * self.pix_size), int(x / 0.01 * self.pix_size)))

		weight_map = depth_map.transpose((1, 0))

		return weight_map


	def set_add_electrode(self, black_x_offset=0.125, black_y_offset=0.055, white_x_offset=0.125, white_y_offset=-0.055):

		robot_x = 0.125
		robot_y = 0.055


		robot_ele = utils.xyz_to_pix([robot_x, robot_y, 0], self.bounds, 0.01)


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

		power_black_ele = utils.xyz_to_pix([self.bounds[0, 1] - black_x_offset, black_y_offset, 0], self.bounds, 0.01)
		power_white_ele = utils.xyz_to_pix([self.bounds[0, 1] - white_x_offset, white_y_offset, 0], self.bounds, 0.01)


		self.analyzer.set_map(self.init_weight_map())

		self.analyzer.set_pathplan(0, [robot_ele[0], robot_ele[1]],
		                           [power_black_ele[0], power_black_ele[1]])

		self.analyzer.set_pathplan(1, [60 - robot_ele[0], robot_ele[1]],
		                           [power_white_ele[0], power_white_ele[1]])


	def add_nono_area(self, analyzer, top_left, bottom_right):
		""" 1:1 pixel location"""

		length = bottom_right[0] - top_left[0] + 1
		width = bottom_right[1] - top_left[1] + 1

		center_x = (bottom_right[0] + top_left[0]) / 2
		center_y = (bottom_right[1] + top_left[1]) / 2

		utils.create_obj(p.GEOM_BOX,
		           mass=-1,
		           halfExtents=[length / 2, width / 2, 0.0001],
		           rgbaColor=[1, 0, 0, 1],
		           basePosition=[center_x, center_y, 0.01],
		           baseOrientation=[0, 0, 0, 1],
                   object_list=self.nono_area
		           )

		top_left_pix = utils.xyz_to_pix([top_left[0], top_left[1], 0], self.bounds, self.pix_size)

		ob_list = []

		for i in range(int(length*100)):
			for j in range(int(width*100)):
				point = (top_left_pix[0] + j, top_left_pix[1] + i)
				ob_list.append(point)

		analyzer.set_obstacles(ob_list)



	def done(self):
		"""Check if the task is done or has failed.

		Returns:
			True if the episode should be considered a success, which we
				use for measuring successes, which is particularly helpful for tasks
				where one may get successes on the very last time step, e.g., getting
				the cloth coverage threshold on the last alllowed action.
				However, for bag-items-easy and bag-items-hard (which use the
				'bag-items' metric), it may be necessary to filter out demos that did
				not attain sufficiently high reward in external code. Currently, this
				is done in `main.py` and its ignore_this_demo() method.
		"""

		# # For tasks with self.metric == 'pose'.
		# if hasattr(self, 'goal'):
		# goal_done = len(self.goal['steps']) == 0  # pylint:
		# disable=g-explicit-length-test
		# return (len(self.goals) == 0) or (self._rewards > 0.99)  # pylint: disable=g-explicit-length-test

		return False  # pylint: disable=g-explicit-length-test

	# return zone_done or defs_done or goal_done

	#-------------------------------------------------------------------------
	# Environment Helper Functions
	#-------------------------------------------------------------------------


	def get_true_image(self, env):
		"""Get RGB-D orthographic heightmaps and segmentation masks."""

		# Capture near-orthographic RGB-D images and segmentation masks.
		color, depth, segm = env.render_camera(self.oracle_cams[0])

		# Combine color with masks for faster processing.
		color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

		# Reconstruct real orthographic projection from point clouds.
		hmaps, cmaps = utils.reconstruct_heightmaps(
				[color], [depth], self.oracle_cams, self.bounds, self.pix_size)

		# Split color back into color and masks.
		cmap = np.uint8(cmaps)[0, Ellipsis, :3]
		hmap = np.float32(hmaps)[0, Ellipsis]
		mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
		return cmap, hmap, mask

	def get_random_pose(self, env, obj_size):
		"""Get random collision-free object pose within workspace bounds."""

		# Get erosion size of object in pixels.
		max_size = np.sqrt(obj_size[0]**2 + obj_size[1]**2)
		erode_size = int(np.round(max_size / self.pix_size))

		_, hmap, obj_mask = self.get_true_image(env)

		# Randomly sample an object pose within free-space pixels.
		free = np.ones(obj_mask.shape, dtype=np.uint8)
		for obj_ids in env.obj_ids.values():
			for obj_id in obj_ids:
				free[obj_mask == obj_id] = 0
		free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
		free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
		if np.sum(free) == 0:
			return None, None
		pix = utils.sample_distribution(np.float32(free))
		pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
		pos = (pos[0], pos[1], obj_size[2] / 2)
		theta = np.random.rand() * 2 * np.pi
		rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
		return pos, rot

	#-------------------------------------------------------------------------
	# Helper Functions
	#-------------------------------------------------------------------------

	def fill_template(self, template, replace):
		"""Read a file and replace key strings."""
		full_template_path = os.path.join(self.assets_root, template)
		with open(full_template_path, 'r') as file:
			fdata = file.read()
		for field in replace:
			for i in range(len(replace[field])):
				fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
		alphabet = string.ascii_lowercase + string.digits
		rname = ''.join(random.choices(alphabet, k=16))
		tmpdir = tempfile.gettempdir()
		template_filename = os.path.split(template)[-1]
		fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
		with open(fname, 'w') as file:
			file.write(fdata)
		return fname

	def get_random_size(self, min_x, max_x, min_y, max_y, min_z, max_z):
		"""Get random box size."""
		size = np.random.rand(3)
		size[0] = size[0] * (max_x - min_x) + min_x
		size[1] = size[1] * (max_y - min_y) + min_y
		size[2] = size[2] * (max_z - min_z) + min_z
		return tuple(size)

	def get_object_points(self, obj):
		obj_shape = p.getVisualShapeData(obj)
		obj_dim = obj_shape[0][3]
		xv, yv, zv = np.meshgrid(
				np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
				np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
				np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
				sparse=False, indexing='xy')
		return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

	def color_random_brown(self, obj):
		shade = np.random.rand() + 0.5
		color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
		p.changeVisualShape(obj, -1, rgbaColor=color)

	def set_assets_root(self, assets_root):
		self.assets_root = assets_root