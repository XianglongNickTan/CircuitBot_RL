import time

import pybullet as p
from utils import utils
import pybullet_data
import cv2
import os, sys
import random
import numpy as np

from tasks import cameras
from utils.pathplanning.pathanalyzer import PathAnalyzer


analyzer = PathAnalyzer()


agent_cams = cameras.RealSenseD415.CONFIG
# agent_cams = cameras.Oracle.CONFIG

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




p.connect(p.GUI)
p.setGravity(0, 0, -10)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setPhysicsEngineParameter(enableFileCaching=0)


p.loadURDF("plane.urdf")

### create plate ###
utils.create_obj(p.GEOM_BOX,
                 mass=-1,
                 halfExtents=[0.4, 0.3, 0.005],
                 rgbaColor=[1, 0.90, 0.72, 1],
                 basePosition=[0.5, 0, 0.005],
                 baseOrientation=[0, 0, 0, 1]
                 )


utils.create_obj(p.GEOM_BOX,
                 mass=-1,
                 halfExtents=[0.01, 0.3, 0.001],
                 rgbaColor=[0, 0, 0, 1],
                 basePosition=[0.1, 0, 0.0052],
                 baseOrientation=[0, 0, 0, 1]
                 )


utils.create_obj(p.GEOM_BOX,
                 mass=-1,
                 halfExtents=[0.01, 0.3, 0.001],
                 rgbaColor=[0, 0, 0, 1],
                 basePosition=[0.9, 0, 0.0052],
                 baseOrientation=[0, 0, 0, 1]
                 )

utils.create_obj(p.GEOM_BOX,
                 mass=-1,
                 halfExtents=[0.4, 0.01, 0.001],
                 rgbaColor=[0, 0, 0, 1],
                 basePosition=[0.5, 0.3, 0.0052],
                 baseOrientation=[0, 0, 0, 1]
                 )

utils.create_obj(p.GEOM_BOX,
                 mass=-1,
                 halfExtents=[0.4, 0.01, 0.001],
                 rgbaColor=[0, 0, 0, 1],
                 basePosition=[0.5, -0.3, 0.0052],
                 baseOrientation=[0, 0, 0, 1]
                 )

utils.create_obj(p.GEOM_MESH,
                 mass=0.01,
                 use_file=OBJECTS['cuboid2'],
                 rgbaColor=utils.COLORS['red'],
                 basePosition=[0.3 + 4 * random.random() / 10,
                               0.10 * (2 * random.random() - 1), 0.02],
                 baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]),
                 )



def render_camera(config):
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


	# Get depth image.
	depth_image_size = (config['image_size'][0], config['image_size'][1])
	zbuffer = np.array(depth).reshape(depth_image_size)
	depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
	depth = (2. * znear * zfar) / depth

	# Get segmentation image.
	segm = np.uint8(segm).reshape(depth_image_size)

	return color, depth, segm


for _ in range(100):
	p.stepSimulation()


for _ in range(10000):
	color, depth, msk = render_camera(agent_cams[2])
	dsize = (80, 80)

	depth = cv2.resize(depth, dsize)
	analyzer.set_map(depth)
	# analyzer.draw_map_3D_only()
	cv2.imshow('test', color)
	cv2.waitKey(1)
	p.stepSimulation()
	time.sleep(1/240)
