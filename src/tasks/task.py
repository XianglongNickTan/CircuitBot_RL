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
from ravens.utils import utils

import pybullet as p

import sys


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

		self.env = env

		self.obj_type = OBJECTS

		# Workspace bounds.
		self.pix_size = 0.005
		self.bounds = np.array([[0.1, 0.9], [-0.28, 0.28], [0, 0.3]])

		self.goals = []
		self.progress = 0
		self._rewards = 0

		self.assets_root = None

		self.objects = []
		self.pick_threshold = 0.03  ## m

		self.grip_z_offset = 0.07

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