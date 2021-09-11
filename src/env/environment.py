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

"""Environment class."""

import os
import pkgutil
import sys
import tempfile
import time

import gym
import numpy as np
from tasks import cameras
from ravens.utils import pybullet_utils
from utils import utils

import pybullet as p


JACO_WORKSPACE_URDF_PATH = 'ur5/workspace.urdf'
PLANE_URDF_PATH = 'plane/plane.urdf'

assets_root = '/assets'

class Environment(gym.Env):
	"""OpenAI Gym-style environment class."""

	def __init__(self,
							 assets_root=assets_root,
							 task=None,
							 disp=False,
							 shared_memory=False,
							 hz=240,
							 use_egl=False):
		"""Creates OpenAI Gym-style environment with PyBullet.

		Args:
			assets_root: root directory of assets.
			task: the task to use. If None, the user must call set_task for the
				environment to work properly.
			disp: show environment with PyBullet's built-in display viewer.
			shared_memory: run with shared memory.
			hz: PyBullet physics simulation step speed. Set to 480 for deformables.
			use_egl: Whether to use EGL rendering. Only supported on Linux. Should get
				a significant speedup in rendering when using.

		Raises:
			RuntimeError: if pybullet cannot load fileIOPlugin.
		"""
		if use_egl and disp:
			raise ValueError('EGL rendering cannot be used with `disp=True`.')

		# self.pix_size = 0.005
		self.pix_size = 0.0025
		self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
		self.agent_cams = cameras.RealSenseD415.CONFIG

		self.assets_root = assets_root

		color_tuple = [
				gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
				for config in self.agent_cams
		]
		depth_tuple = [
				gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
				for config in self.agent_cams
		]
		self.observation_space = gym.spaces.Dict({
				'color': gym.spaces.Tuple(color_tuple),
				'depth': gym.spaces.Tuple(depth_tuple),
		})
		self.position_bounds = gym.spaces.Box(
				low=np.array([0.4, -0.25, 0.], dtype=np.float32),
				high=np.array([0.8, 0.25, 0.28], dtype=np.float32),
				shape=(3,),
				dtype=np.float32)
		self.action_space = gym.spaces.Dict({
				'pose0':
						gym.spaces.Tuple(
								(self.position_bounds,
								 gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
				'pose1':
						gym.spaces.Tuple(
								(self.position_bounds,
								 gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
		})

		# Start PyBullet.
		disp_option = p.DIRECT
		if disp:
			disp_option = p.GUI
			if shared_memory:
				disp_option = p.SHARED_MEMORY
		client = p.connect(disp_option)
		file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
		if file_io < 0:
			raise RuntimeError('pybullet: cannot load FileIO!')
		if file_io >= 0:
			p.executePluginCommand(
					file_io,
					textArgument=assets_root,
					intArgs=[p.AddFileIOAction],
					physicsClientId=client)

		self._egl_plugin = None
		if use_egl:
			assert sys.platform == 'linux', ('EGL rendering is only supported on '
																			 'Linux.')
			egl = pkgutil.get_loader('eglRenderer')
			if egl:
				self._egl_plugin = p.loadPlugin(egl.get_filename(),
																				'_eglRendererPlugin')
			else:
				self._egl_plugin = p.loadPlugin('eglRendererPlugin')
			print('EGL renderering enabled.')

		p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
		p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
		p.setPhysicsEngineParameter(enableFileCaching=0)
		p.setAdditionalSearchPath(assets_root)
		p.setAdditionalSearchPath(tempfile.gettempdir())
		p.setTimeStep(1. / hz)

		# If using --disp, move default camera closer to the scene.
		if disp:
			target = p.getDebugVisualizerCamera()[11]
			p.resetDebugVisualizerCamera(
					cameraDistance=1.3,
					cameraYaw=90,
					cameraPitch=-25,
					cameraTargetPosition=target)


		self.init_sim()

		if task:
			self.set_task(task)

	def init_sim(self):
		pybullet_utils.load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH),
														 [0, 0, -0.001])

		utils.create_obj(p.GEOM_BOX,
						mass=-1,
						halfExtents=[0.4, 0.3, 0.005],
						rgbaColor=[1, 0.90, 0.72, 1],
						basePosition=[0.5, 0, 0.005],
						baseOrientation=[0, 0, 0, 1]
						)


	def is_static(self, object):
		"""Return true if objects are no longer moving."""
		v = [np.linalg.norm(p.getBaseVelocity(i)[0])
			 for i in object]
		return all(np.array(v) < 5e-3)

	def add_object(self, urdf, pose, category='rigid'):
		"""List of (fixed, rigid, or deformable) objects in env."""
		fixed_base = 1 if category == 'fixed' else 0
		obj_id = pybullet_utils.load_urdf(
				p,
				os.path.join(self.assets_root, urdf),
				pose[0],
				pose[1],
				useFixedBase=fixed_base)
		self.obj_ids[category].append(obj_id)
		return obj_id

	#---------------------------------------------------------------------------
	# Standard Gym Functions
	#---------------------------------------------------------------------------

	def seed(self, seed=None):
		self._random = np.random.RandomState(seed)
		return seed


	def reset(self):
		"""Performs common reset functionality for all supported tasks."""
		if not self.task:
			raise ValueError('environment task must be set. Call set_task or pass '
											 'the task arg in the environment constructor.')
		self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
		p.setGravity(0, 0, -9.8)

		# Temporarily disable rendering to load scene faster.
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)


		# Reset task.
		self.task.reset()

		# Re-enable rendering.
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

		obs, _, _, _ = self.step()
		return obs



	def step(self, action=None):
		"""Execute action with specified primitive.

		Args:
			action: action to execute.

		Returns:
			(obs, reward, done, info) tuple containing MDP step data.
		"""
		if action is not None:
			self.task.apply_action(action)


		# Step simulator asynchronously until objects settle.
		while not self.is_static(self.task.objects):
			p.stepSimulation()

		# Get task rewards.
		reward, info = self.task.reward() if action is not None else (0, {})
		done = self.task.done()

		# Add ground truth robot state into info.
		# info.update(self.info)

		obs = self._get_obs()

		return obs, reward, done, info

	def close(self):
		if self._egl_plugin is not None:
			p.unloadPlugin(self._egl_plugin)
		p.disconnect()

	def render(self, mode='rgb_array'):
		# Render only the color image from the first camera.
		# Only support rgb_array for now.
		if mode != 'rgb_array':
			raise NotImplementedError('Only rgb_array implemented')
		color, _, _ = self.render_camera(self.agent_cams[0])
		return color

	def render_camera(self, config):
		"""Render RGB-D image with specified camera configuration."""

		# OpenGL camera settings.
		lookdir = np.float32([0, 0, 1]).reshape(3, 1)
		updir = np.float32([0, -1, 0]).reshape(3, 1)
		rotation = p.getMatrixFromQuaternion(config['rotation'])
		rotm = np.float32(rotation).reshape(3, 3)
		lookdir = (rotm @ lookdir).reshape(-1)
		updir = (rotm @ updir).reshape(-1)
		lookat = config['position'] + lookdir
		focal_len = config['intrinsics'][0]
		znear, zfar = config['zrange']
		viewm = p.computeViewMatrix(config['position'], lookat, updir)
		fovh = (config['image_size'][0] / 2) / focal_len
		fovh = 180 * np.arctan(fovh) * 2 / np.pi

		# Notes: 1) FOV is vertical FOV 2) aspect must be float
		aspect_ratio = config['image_size'][1] / config['image_size'][0]
		projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

		# Render with OpenGL camera settings.
		_, _, color, depth, segm = p.getCameraImage(
				width=config['image_size'][1],
				height=config['image_size'][0],
				viewMatrix=viewm,
				projectionMatrix=projm,
				shadow=1,
				flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
				# Note when use_egl is toggled, this option will not actually use openGL
				# but EGL instead.
				renderer=p.ER_BULLET_HARDWARE_OPENGL)

		# Get color image.
		color_image_size = (config['image_size'][0], config['image_size'][1], 4)
		color = np.array(color, dtype=np.uint8).reshape(color_image_size)
		color = color[:, :, :3]  # remove alpha channel
		if config['noise']:
			color = np.int32(color)
			color += np.int32(self._random.normal(0, 3, config['image_size']))
			color = np.uint8(np.clip(color, 0, 255))

		# Get depth image.
		depth_image_size = (config['image_size'][0], config['image_size'][1])
		zbuffer = np.array(depth).reshape(depth_image_size)
		depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
		depth = (2. * znear * zfar) / depth
		if config['noise']:
			depth += self._random.normal(0, 0.003, depth_image_size)

		# Get segmentation image.
		segm = np.uint8(segm).reshape(depth_image_size)

		return color, depth, segm

	@property
	def info(self):
		"""Environment info variable with object poses, dimensions, and colors."""

		# Some tasks create and remove zones, so ignore those IDs.
		# removed_ids = []
		# if (isinstance(self.task, tasks.names['cloth-flat-notarget']) or
		#         isinstance(self.task, tasks.names['bag-alone-open'])):
		#   removed_ids.append(self.task.zone_id)

		info = {}  # object id : (position, rotation, dimensions)
		for obj_ids in self.obj_ids.values():
			for obj_id in obj_ids:
				pos, rot = p.getBasePositionAndOrientation(obj_id)
				dim = p.getVisualShapeData(obj_id)[0][3]
				info[obj_id] = (pos, rot, dim)
		return info

	def set_task(self, task):
		task.set_assets_root(self.assets_root)
		self.task = task

	#---------------------------------------------------------------------------
	# Robot Movement Functions
	#---------------------------------------------------------------------------


	def _get_obs(self):
		# Get RGB-D camera image observations.
		obs = {'color': (), 'depth': ()}
		for config in self.agent_cams:
			color, depth, _ = self.render_camera(config)
			obs['color'] += (color,)
			obs['depth'] += (depth,)

		return obs

