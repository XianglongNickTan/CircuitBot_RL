import pybullet as p
import time
import cv2
import numpy as np
import pybullet_data
from gym import spaces, Env
import sys


from tasks.cameras import DaBai
from utils import utils

# from tasks.cameras import RealSenseD415

sys.path.insert(1, "../bullet3/build_cmake/examples/pybullet")
timeStep = 1 / 240.0


np.set_printoptions(precision=2, floatmode='fixed', suppress=True)


class Environment(Env):
	def __init__(self,
				 done_after=10000,
				 disp=True,
				 hz=240,
				 shared_memory=False,
				 use_egl=False,
				 task=None
				 ):

		### pybullet setting ###
		if disp:
			physics_client = p.connect(p.GUI)
		else:
			physics_client = p.connect(p.DIRECT)

		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

		p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
		p.setPhysicsEngineParameter(enableFileCaching=0)
		p.setTimeStep(1. / hz)

		### map settings ###
		### x - row - height
		### y - colomn - widthz



		### robot pick place settings ###
		self.reach_x = [0.30, 0.68]
		self.reach_y = [-0.2, 0.2]
		self.reach_z = [0.06, 0.4]


		### training settings ###
		self.numSteps = 0
		self.doneAfter = done_after


		### print env settings ###

		### fake camera ###
		self.agent_cams = DaBai.CONFIG

		# color_tuple = [
		#     spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
		#     for config in self.agent_cams
		# ]
		# depth_tuple = [
		#     spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
		#     for config in self.agent_cams
		# ]
		#
		# self.observation_space = spaces.Dict({
		#     'color': spaces.Tuple(color_tuple),
		#     'depth': spaces.Tuple(depth_tuple),
		# })


		self.position_bounds = spaces.Box(
			low=np.array([0.25, -0.5, 0.], dtype=np.float32),
			high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
			shape=(3,),
			dtype=np.float32)
		self.action_space = spaces.Dict({
			'pose0':
				spaces.Tuple(
					(self.position_bounds,
					 spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
			'pose1':
				spaces.Tuple(
					(self.position_bounds,
					 spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
		})







		### action = [x, y, x, y, ori] ###
		# self.action_space = spaces.Box(
		# 	low=np.array([22*self.pixel_ratio, 8*self.pixel_ratio, 22*self.pixel_ratio, 8*self.pixel_ratio, 0]) ,
		# 	high=np.array([58*self.pixel_ratio, 47*self.pixel_ratio, 58*self.pixel_ratio, 47*self.pixel_ratio, 1]),
		# 	dtype=np.int)

		p.setGravity(0, 0, -10)


		### init map ###
		self.init_sim()


		if task:
			self.set_task(task)


		self.objects = []



	def set_task(self, task):
		self.task = task




	def close(self):
		p.disconnect()


	def is_static(self, object):
		"""Return true if objects are no longer moving."""
		v = [np.linalg.norm(p.getBaseVelocity(i)[0])
			 for i in object]
		return all(np.array(v) < 5e-3)


	def seed(self, seed=None):
		self._random = np.random.RandomState(seed)
		return seed

	def render(self, mode='rgb_array'):
		# Render only the color image from the first camera.
		# Only support rgb_array for now.
		if mode != 'rgb_array':
			raise NotImplementedError('Only rgb_array implemented')
		color, _, _ = self.render_camera(self.agent_cams[0])
		return color


	def render_camera(self, config):
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
			shadow=0,
			flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
			# Note when use_egl is toggled, this option will not actually use openGL
			# but EGL instead.
			renderer=p.ER_BULLET_HARDWARE_OPENGL)


		width_clip = int(24 * (config['image_size'][0] / 160))

		# Get color image.
		color_image_size = (config['image_size'][0], config['image_size'][1], 4)
		color = np.array(color, dtype=np.uint8).reshape(color_image_size)
		color = color[:, :, :3]  # remove alpha channel
		if config['noise']:
			color = np.int32(color)
			color += np.int32(self._random.normal(0, 3, config['image_size']))
			color = np.uint8(np.clip(color, 0, 255))
		# color = color[:, width_clip:-width_clip, :]

		# Get depth image.
		depth_image_size = (config['image_size'][0], config['image_size'][1])
		zbuffer = np.array(depth).reshape(depth_image_size)
		# depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
		# depth = (2. * znear * zfar) / depth
		depth = zfar - zfar * znear / (zfar - (zfar - znear) * zbuffer)

		if config['noise']:
			depth += self._random.normal(0, 0.003, depth_image_size)
		# depth = depth[:, width_clip:-width_clip]

		# Get segmentation image.
		segm = np.uint8(segm).reshape(depth_image_size)
		# segm = segm[:, width_clip:-width_clip]

		depth_image_size = (config['image_size'][0], config['image_size'][1], 1)

		depth.reshape(depth_image_size)

		rgb_d = np.dstack((color, depth))

		cv2.imshow('test', color)
		cv2.waitKey(1)

		return color, depth, segm, rgb_d





	def init_sim(self):

		p.loadURDF("plane.urdf")

		### create plate ###
		utils.create_obj(p.GEOM_BOX,
						mass=-1,
						halfExtents=[0.4, 0.28, 0.005],
						rgbaColor=[1, 0.90, 0.72, 1],
						basePosition=[0.5, 0, 0.005],
						baseOrientation=[0, 0, 0, 1]
						)



	# def _get_obs(self):
	# 	# Get RGB-D camera image observations.
	# 	obs = {'color': (), 'depth': ()}
	# 	color, depth, _, rgb_d = self.render_camera(self.agent_cams[0])
	# 	obs['color'] += (color,)
	# 	obs['depth'] += (depth,)
	#
	# 	return obs, depth


	def _get_obs(self):
		# Get RGB-D camera image observations.
		obs = {'color': (), 'depth': ()}
		color, depth, _, rgb_d = self.render_camera(self.agent_cams[0])
		obs['color'] += (color,)
		obs['depth'] += (depth,)

		return rgb_d, depth


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



	def reset(self):

		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

		self.task.reset()

		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

		# obs, _, _, _ = self.step()

		for _ in range(100):
			p.stepSimulation()

		obs, _ = self._get_obs()

		return obs


	def step(self, action=None):
		""" Execute one time step within the environment."""


		if action is not None:
			self.task.apply_action(action)


		while not self.is_static(self.task.objects):
			p.stepSimulation()

		obs, depth = self._get_obs()


		reward = self.task.reward(depth) if action is not None else (0, {})
		done = self.task.done()

		info = {}

		if done:
			info = {"episode": {"l": self.numSteps, "r": reward}}



		if self.numSteps == 0:
			self.startTime = time.time()

		self.numSteps += 1

		return obs, reward, done, info
