import pybullet as p
import time
import cv2
import numpy as np
import pybullet_data
from gym import spaces, Env
import sys
import math


from circuitbot.jaco_sim.jaco import Jaco
from tasks.cameras import DaBai
# from tasks.cameras import RealSenseD415

sys.path.insert(1, "../bullet3/build_cmake/examples/pybullet")
timeStep = 1 / 240.0


np.set_printoptions(precision=2, floatmode='fixed', suppress=True)


class Environment(Env):
	def __init__(self,
				 map_row=80,
				 map_column=56,
				 n_substeps=5,  # Number of simulation steps to do in every env step.
				 done_after=10000,
				 workspace_width=56,
				 workspace_height=80,
				 plate_offset=10,
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
		self.plate_offset = plate_offset
		self.row = map_row
		self.column = map_column

		self.workspace_width = workspace_width
		self.workspace_height = workspace_height


		### robot pick place settings ###
		self.reach_x = [0.30, 0.68]
		self.reach_y = [-0.2, 0.2]
		self.reach_z = [0.06, 0.4]


		### training settings ###
		self.numSteps = 0
		self.n_substeps = n_substeps
		self.doneAfter = done_after
		self.pixel_ratio = 2        ### increase picture size by 2
		self.weight_map_ratio = 1   ### 1:1 weight map

		### print env settings ###
		self.electrode_x_offset = 15  ### row in weight map
		self.electrode_y_offset = 6

		self.ele_r_n = int(self.workspace_height - self.electrode_x_offset)
		self.ele_r_f = int(self.electrode_x_offset - 1)
		self.ele_c_l = int(self.workspace_width / 2 + self.electrode_y_offset - 1)
		self.ele_c_r = int(self.workspace_width / 2 - self.electrode_y_offset)

		self.ele_n_l = self.from_pixel_to_coordinate([self.ele_r_n, self.ele_c_l], self.weight_map_ratio)
		self.ele_n_r = self.from_pixel_to_coordinate([self.ele_r_n, self.ele_c_r], self.weight_map_ratio)
		self.ele_f_l = self.from_pixel_to_coordinate([self.ele_r_f, self.ele_c_l], self.weight_map_ratio)
		self.ele_f_r = self.from_pixel_to_coordinate([self.ele_r_f, self.ele_c_r], self.weight_map_ratio)


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
		self.action_space = spaces.Box(
			low=np.array([22*self.pixel_ratio, 8*self.pixel_ratio, 22*self.pixel_ratio, 8*self.pixel_ratio, 0]) ,
			high=np.array([58*self.pixel_ratio, 47*self.pixel_ratio, 58*self.pixel_ratio, 47*self.pixel_ratio, 1]),
			dtype=np.int)

		self.observation_space = spaces.Box(
			-np.inf, np.inf, shape=(self.row, self.column), dtype='float32')




		if task:
			self.set_task(task)


		self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
		self.objects = []


		### init map ###
		self._init_sim()

		### init jaco ###
		self.arm = Jaco()


	def close(self):
		p.disconnect()


	@property
	def is_static(self):
		"""Return true if objects are no longer moving."""
		v = [np.linalg.norm(p.getBaseVelocity(i)[0])
			 for i in self.obj_ids['rigid']]
		return all(np.array(v) < 5e-3)

	def add_object(self, urdf, pose, category='rigid'):
		"""List of (fixed, rigid, or deformable) objects in env."""
		fixed_base = 1 if category == 'fixed' else 0
		obj_id = None
		self.obj_ids[category].append(obj_id)
		return obj_id

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
		projm = p.computeProjectionMatrixFOV(fovh, 1, znear, zfar)

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


		width_clip = int(24 * (self.pixel_ratio / 2))

		# Get color image.
		color_image_size = (config['image_size'][0], config['image_size'][1], 4)
		color = np.array(color, dtype=np.uint8).reshape(color_image_size)
		color = color[:, :, :3]  # remove alpha channel
		if config['noise']:
			color = np.int32(color)
			color += np.int32(self._random.normal(0, 3, config['image_size']))
			color = np.uint8(np.clip(color, 0, 255))
		color = color[:, width_clip:-width_clip, :]

		# Get depth image.
		depth_image_size = (config['image_size'][0], config['image_size'][1])
		zbuffer = np.array(depth).reshape(depth_image_size)
		depth = zfar - zfar * znear / (zfar - (zfar - znear) * zbuffer)

		if config['noise']:
			depth += self._random.normal(0, 0.003, depth_image_size)
		depth = depth[:, width_clip:-width_clip]

		# Get segmentation image.
		segm = np.uint8(segm).reshape(depth_image_size)
		segm = segm[:, width_clip:-width_clip]


		return color, depth, segm




	def create_obj(self, obj, mass=None, halfExtents=None, radius=None, height=None, rgbaColor=None,
				   basePosition=None, baseOrientation=None, use_file=None):

		if not use_file:

			if obj == p.GEOM_BOX:
				visual = p.createVisualShape(obj, halfExtents=halfExtents, rgbaColor=rgbaColor)
				shape = p.createCollisionShape(obj, halfExtents=halfExtents)

			elif obj == p.GEOM_CYLINDER:
				# visual = p.createVisualShape(obj, radius=radius, length=height, rgbaColor=rgbaColor)
				visual = p.createVisualShape(obj, radius=radius, length=height, rgbaColor=rgbaColor)
				# shape = p.createCollisionShape(obj, radius=radius, height=height)
				shape = -1

			else:
				raise NotImplementedError()

		else:
			visual = p.createVisualShape(obj, fileName=use_file, rgbaColor=rgbaColor)
			shape = p.createCollisionShape(obj, fileName=use_file)

		objID = p.createMultiBody(baseMass=mass,
								  baseCollisionShapeIndex=shape,
								  baseVisualShapeIndex=visual,
								  basePosition=basePosition,
								  baseOrientation=baseOrientation)

		return objID


	def _init_sim(self):

		p.loadURDF("plane.urdf")

		### create plate ###
		self.create_obj(p.GEOM_BOX,
						mass=-1,
						halfExtents=[self.workspace_height/200, self.workspace_width/200, 0.005],
						rgbaColor=[1, 0.90, 0.72, 1],
						basePosition=[self.workspace_height/200+0.1, 0, 0.005],
						baseOrientation=[0, 0, 0, 1]
						)

		### create near electrode ###
		self.create_obj(p.GEOM_BOX,
						mass=-1,
						halfExtents=[0.0075, 0.0075, 0.0001],
						rgbaColor=[0, 0, 0, 1],
						basePosition=[self.ele_n_l[0], self.ele_n_l[1], 0.01],
						baseOrientation=[0, 0, 0, 1]
						)

		self.create_obj(p.GEOM_BOX,
						mass=-1,
						halfExtents=[self.electrode_x_offset / 200, 0.005, 0.0001],
						rgbaColor=[0, 0, 0, 1],
						basePosition=[(self.ele_n_l[0] + 0.1) / 2, self.ele_n_l[1], 0.01],
						baseOrientation=[0, 0, 0, 1]
						)

		self.create_obj(p.GEOM_BOX,
						mass=-1,
						halfExtents=[0.0075, 0.0075, 0.0001],
						rgbaColor=[1, 1, 1, 1],
						basePosition=[self.ele_n_r[0], self.ele_n_r[1], 0.01],
						baseOrientation=[0, 0, 0, 1]
						)

		self.create_obj(p.GEOM_BOX,
						mass=-1,
						halfExtents=[self.electrode_x_offset / 200, 0.005, 0.0001],
						rgbaColor=[1, 1, 1, 1],
						basePosition=[(self.ele_n_r[0] + 0.1) / 2, self.ele_n_r[1], 0.01],
						baseOrientation=[0, 0, 0, 1]
						)


		### create far electrode ###
		self.create_obj(p.GEOM_BOX,
						mass=-1,
						halfExtents=[0.0075, 0.0075, 0.0001],
						rgbaColor=[0, 0, 0, 1],
						basePosition=[self.ele_f_l[0], self.ele_f_l[1], 0.01],
						baseOrientation=[0, 0, 0, 1]
						)


		self.create_obj(p.GEOM_BOX,
						mass=-1,
						halfExtents=[self.electrode_x_offset / 200, 0.005, 0.0001],
						rgbaColor=[0, 0, 0, 1],
						basePosition=[self.ele_f_l[0] + self.electrode_x_offset / 200, self.ele_f_l[1], 0.01],
						baseOrientation=[0, 0, 0, 1]
						)

		self.create_obj(p.GEOM_BOX,
						mass=-1,
						halfExtents=[0.0075, 0.0075, 0.0001],
						rgbaColor=[1, 1, 1, 1],
						basePosition=[self.ele_f_r[0], self.ele_f_r[1], 0.01],
						baseOrientation=[0, 0, 0, 1]
						)


		self.create_obj(p.GEOM_BOX,
						mass=-1,
						halfExtents=[self.electrode_x_offset / 200, 0.005, 0.0001],
						rgbaColor=[1, 1, 1, 1],
						basePosition=[self.ele_f_r[0] + self.electrode_x_offset / 200, self.ele_f_r[1], 0.01],
						baseOrientation=[0, 0, 0, 1]
						)




	def from_pixel_to_coordinate(self, x_y, ratio):
		""" ratio: 1 = 1:1 cm  2 = 1:0.5cm """
		real_x = self.workspace_height - (0.5 + x_y[0]) / ratio
		real_y = self.workspace_width / 2 - (0.5 + x_y[1]) / ratio

		return [(real_x + self.plate_offset) / 100, real_y / 100]




	def _get_obs(self):
		# Get RGB-D camera image observations.
		obs = {'color': (), 'depth': ()}
		color, depth, _ = self.render_camera(self.agent_cams[0])
		obs['color'] += (color,)
		obs['depth'] += (depth,)

		return obs



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
		self.task = task



	def reset(self):


		p.setGravity(0, 0, -10)


		self.task.reset()

		obs, _, _, _ = self.step()

		return obs


	def step(self, action=None):
		""" Execute one time step within the environment."""

		# if action is not None:
		#     self.task.apply_action(action)

		if action is not None:
			self.task.apply_action(action)


		while not self.is_static:
			p.stepSimulation()

		reward = self.task.reward() if action is not None else (0, {})
		done = self.task.done()

		info = {}

		if done:
			info = {"episode": {"l": self.numSteps, "r": reward}}

		obs = self._get_obs()

		if self.numSteps == 0:
			self.startTime = time.time()

		self.numSteps += 1

		return obs, reward, done, info
