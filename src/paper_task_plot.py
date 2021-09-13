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
from circuitbot.jaco_sim.jaco import Jaco

analyzer = PathAnalyzer()


agent_cams = cameras.Paper_plot.CONFIG
# agent_cams = cameras.Oracle.CONFIG
pix_size = 0.0025

bounds = np.array([[0.1, 0.9], [-0.3, 0.3], [0, 0.3]])

rootdir = os.path.dirname(sys.modules['__main__'].__file__)
rootdir += "/assets/blocks"


obj_cube = rootdir + "/cube_4.obj"
obj_cuboid1 = rootdir + "/cuboid_4_4_8.obj"
obj_cuboid2 = rootdir + "/cuboid_4_16.obj"
obj_cuboid3 = rootdir + "/cuboid_8_8_4.obj"
obj_curve = rootdir + "/curve.obj"
obj_cylinder = rootdir + "/cylinder_4_4.obj"
obj_triangular_prism = rootdir + "/triangular_prism_4_8.obj"
obj_bridge = rootdir + "/bridge_1.obj"
# obj_bridge = rootdir + "/bridge_2.obj"


object_list = []

OBJECTS = {
	'cube': obj_cube,
	'cuboid1': obj_cuboid1,
	'cuboid2': obj_cuboid2,
	'cuboid3': obj_cuboid3,
	'curve': obj_curve,
	'cylinder': obj_cylinder,
	'triangular_prism': obj_triangular_prism,
	'bridge': obj_bridge
}

in_shape = (120, 160, 6)

forbidden_area = []
p.connect(p.GUI)
p.setGravity(0, 0, -10)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, lightPosition=[1, -1, 3])
# p.configureDebugVisualizer(p.lightPosition, [1, -1, 4])

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
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

#
# utils.create_obj(p.GEOM_BOX,
#                  mass=-1,
#                  halfExtents=[0.01, 0.3, 0.001],
#                  rgbaColor=[0, 0, 0, 1],
#                  basePosition=[0.1, 0, 0.0052],
#                  baseOrientation=[0, 0, 0, 1]
#                  )
#
#
# utils.create_obj(p.GEOM_BOX,
#                  mass=-1,
#                  halfExtents=[0.01, 0.3, 0.001],
#                  rgbaColor=[0, 0, 0, 1],
#                  basePosition=[0.9, 0, 0.0052],
#                  baseOrientation=[0, 0, 0, 1]
#                  )
#
# utils.create_obj(p.GEOM_BOX,
#                  mass=-1,
#                  halfExtents=[0.4, 0.01, 0.001],
#                  rgbaColor=[0, 0, 0, 1],
#                  basePosition=[0.5, 0.3, 0.0052],
#                  baseOrientation=[0, 0, 0, 1]
#                  )
#
# utils.create_obj(p.GEOM_BOX,
#                  mass=-1,
#                  halfExtents=[0.4, 0.01, 0.001],
#                  rgbaColor=[0, 0, 0, 1],
#                  basePosition=[0.5, -0.3, 0.0052],
#                  baseOrientation=[0, 0, 0, 1]
#                  )
#

def add_obstacles():
	# utils.create_obj(p.GEOM_MESH,
	#                  mass=0.01,
	#                  use_file=obj_cuboid2,
	#                  rgbaColor=[186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0, 1],
	#                  basePosition=[0.5,
	#                                0.0, 0.1],
	#                  baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
	#                  object_list=object_list
	#                  )

	# utils.create_obj(p.GEOM_MESH,
	#                  mass=0.01,
	#                  use_file=obj_cuboid2,
	#                  rgbaColor=[186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0, 1],
	#                  basePosition=[0.22, -0.19, 0.04],
	#                  baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
	#                  object_list=object_list
	#                  )

	# utils.create_obj(p.GEOM_MESH,
	#                  mass=0.01,
	#                  use_file=obj_bridge,
	#                  rgbaColor=[186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0, 1],
	#                  basePosition=[0.3,
	#                                0.15, 0.1],
	#                  baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
	#                  object_list=object_list
	#                  )


	# utils.create_obj(p.GEOM_MESH,
	#                  mass=0.01,
	#                  use_file=obj_bridge,
	#                  rgbaColor=[186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0, 1],
	#                  basePosition=[0.5,
	#                                -0.05, 0.04],
	#                  baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
	#                  object_list=object_list
	#                  )


	# utils.create_obj(p.GEOM_MESH,
	#                  mass=0.01,
	#                  use_file=obj_bridge,
	#                  rgbaColor=[186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0, 1],
	#                  basePosition=[0.3,
	#                                -0.15, 0.1],
	#                  baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]),
	#                  object_list=object_list
	#                  )


	utils.create_obj(p.GEOM_MESH,
	                 mass=0.01,
	                 use_file=obj_bridge,
	                 rgbaColor=[186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0, 1],
	                 basePosition=[0.64,
	                               0, 0.04],
	                 baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/8]),
	                 object_list=object_list
	                 )



def build_bin():
	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[0.005, 0.075, 0.001],
	                 rgbaColor=[078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0, 1],
	                 basePosition=[0.15, -0.20, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )


	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[0.075, 0.005, 0.001],
	                 rgbaColor=[078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0, 1],
	                 basePosition=[0.22, -0.275, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[0.005, 0.075, 0.001],
	                 rgbaColor=[078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0, 1],
	                 basePosition=[0.29, -0.20, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[0.075, 0.005, 0.001],
	                 rgbaColor=[078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0, 1],
	                 basePosition=[0.22, -0.12, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )


def set_add_electrode(black_x_offset=0.06, black_y_offset=0.05, white_x_offset=0.06, white_y_offset=-0.05):
	robot_x = 0.06
	robot_y = 0.05

	robot_ele = np.asarray(utils.xyz_to_pix([robot_x + bounds[0, 0], robot_y, 0], bounds, 0.01))
	robot_ele[1] = 79 - robot_ele[1]

	size = 0.0085

	### create extension electrode ###

	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[0.040, 0.005, 0.0001],
	                 rgbaColor=[0, 0, 0, 1],
	                 basePosition=[0.06, robot_y, 0.0],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[0.005, 0.065, 0.0001],
	                 rgbaColor=[0, 0, 0, 1],
	                 basePosition=[0.02, 0.11, 0.0],
	                 baseOrientation=[0, 0, 0, 1]
	                 )


	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[0.06, 0.005, 0.0001],
	                 rgbaColor=[255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0, 1],
	                 basePosition=[0.05, -robot_y, 0.0],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[0.005, 0.135, 0.0001],
	                 rgbaColor=[255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0, 1],
	                 basePosition=[-0.01, 0.08, 0.0],
	                 baseOrientation=[0, 0, 0, 1]
	                 )


	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[0.040, 0.005, 0.0001],
	                 rgbaColor=[0,0,0, 1],
	                 basePosition=[0.94, robot_y, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )



	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[0.06, 0.005, 0.0001],
	                 rgbaColor=[255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0, 1],
	                 basePosition=[0.94, -robot_y, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )




	### create robot electrode ###


	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[size, size, 0.0001],
	                 rgbaColor=[0, 0, 0, 1],
	                 basePosition=[robot_x + bounds[0, 0], robot_y, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[robot_x / 2, 0.005, 0.0001],
	                 rgbaColor=[0, 0, 0, 1],
	                 basePosition=[robot_x / 2 + bounds[0, 0], robot_y, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[size, size, 0.0001],
	                 rgbaColor=[255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0, 1],
	                 basePosition=[robot_x + bounds[0, 0], -robot_y, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[robot_x / 2, 0.005, 0.0001],
	                 rgbaColor=[255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0, 1],
	                 basePosition=[robot_x / 2 + bounds[0, 0], -robot_y, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	### create power source electrode ###
	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[size, size, 0.0001],
	                 rgbaColor=[0, 0, 0, 1],
	                 basePosition=[bounds[0, 1] - black_x_offset, black_y_offset, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[black_x_offset / 2, 0.005, 0.0001],
	                 rgbaColor=[0, 0, 0, 1],
	                 basePosition=[bounds[0, 1] - black_x_offset / 2, black_y_offset, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[size, size, 0.0001],
	                 rgbaColor=[255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0, 1],
	                 basePosition=[bounds[0, 1] - white_x_offset, white_y_offset, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	utils.create_obj(p.GEOM_BOX,
	                 mass=-1,
	                 halfExtents=[white_x_offset / 2, 0.005, 0.0001],
	                 rgbaColor=[255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0, 1],
	                 basePosition=[bounds[0, 1] - white_x_offset / 2, white_y_offset, 0.01],
	                 baseOrientation=[0, 0, 0, 1]
	                 )

	power_black_ele = np.asarray(utils.xyz_to_pix([bounds[0, 1] - black_x_offset, black_y_offset, 0],
	                                              bounds, 0.01))

	power_white_ele = np.asarray(utils.xyz_to_pix([bounds[0, 1] - white_x_offset, white_y_offset, 0],
	                                              bounds, 0.01))

	power_black_ele[1] = 79 - power_black_ele[1]
	power_white_ele[1] = 79 - power_white_ele[1]

	analyzer.set_map(np.zeros([80, 60]))

	analyzer.set_pathplan(0, [robot_ele[0], robot_ele[1]],
	                           [power_black_ele[0], power_black_ele[1]])

	analyzer.set_pathplan(1, [60 - robot_ele[0], robot_ele[1]],
	                           [power_white_ele[0], power_white_ele[1]])



	# analyzer.set_pathplan(0, [robot_ele[0], robot_ele[1]],
	#                            [power_black_ele[0], power_black_ele[1]])
	#
	# analyzer.set_pathplan(1, [60 - robot_ele[0], robot_ele[1]],
	#                            [power_white_ele[0], power_white_ele[1]])


	electrode_list = []
	for i in range(78 - robot_ele[1]):
		electrode_list.append([robot_ele[0], robot_ele[1] + i + 2])
		electrode_list.append([60 - robot_ele[0], robot_ele[1] + i + 2])

	for i in range(power_black_ele[1]):
		electrode_list.append([power_black_ele[0], i])

	for i in range(power_white_ele[1]):
		electrode_list.append([power_white_ele[0], i])

	# analyzer.add_obstacles(electrode_list)




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


def _get_obs():
	# Get RGB-D camera image observations.
	obs = {'color': (), 'depth': ()}
	for config in cameras.RealSenseD415.CONFIG:
		color, depth, _ = render_camera(config)
		obs['color'] += (color,)
		obs['depth'] += (depth,)

	return obs


def get_image(obs):
	"""Stack color and height images image."""

	# if use_goal_image:
	#   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
	#   goal_image = concatenate_c_h(colormap_g, heightmap_g)
	#   input_image = np.concatenate((input_image, goal_image), axis=2)
	#   assert input_image.shape[2] == 12, input_image.shape

	# Get color and height maps from RGB-D images.
	cmap, hmap = utils.get_fused_heightmap(
			obs, cameras.RealSenseD415.CONFIG, bounds, pix_size)
	img = np.concatenate((cmap,
												hmap[Ellipsis, None],
												hmap[Ellipsis, None],
												hmap[Ellipsis, None]), axis=2)
	# assert img.shape == in_shape, img.shape
	return img



def get_true_image():
	"""Get RGB-D orthographic heightmaps and segmentation masks."""

	# Capture near-orthographic RGB-D images and segmentation masks.
	color, depth, segm = render_camera(cameras.Oracle.CONFIG[0])

	# Combine color with masks for faster processing.
	color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

	# Reconstruct real orthographic projection from point clouds.
	hmaps, cmaps = utils.reconstruct_heightmaps(
		[color], [depth], agent_cams, bounds, pix_size)

	# Split color back into color and masks.
	cmap = np.uint8(cmaps)[0, Ellipsis, :3]
	hmap = np.float32(hmaps)[0, Ellipsis]
	mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
	return cmap, hmap, mask


def get_weight_map():

	# _, depth_map, _ = get_true_image()

	obs = _get_obs()
	img = get_image(obs)

	depth_map = img[:, :, 3]

	x, y = depth_map.shape[0:2]



	depth_map *= 100    ### convert to cm
	depth_map -= 1      ### minus plate height

	#### resize the map to weight map ######
	depth_map = cv2.resize(depth_map, (int(y / 0.01 * pix_size), int(x / 0.01 * pix_size)))

	weight_map = depth_map.transpose((1, 0))

	return weight_map


def show_path(weight_map, path, no):
	for point in path:
		center_u = point[0]
		center_v = point[1]
		center_v = 79 - center_v

		center = np.asarray(utils.pix_to_xyz(pixel=[center_u, center_v],
		                          height=None,
		                          bounds=bounds,
		                          pixel_size=0.01,
		                          skip_height=True))

		center[2] = 0.4

		center_z = weight_map[center_v, center_u] / 100


		arm.move_to(center,
		                 p.getQuaternionFromEuler([0, -np.pi, 0]))

		if no == 0:
			color = [0,0,0,1]

		if no == 1:
			color = [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0, 1]

		if center_z <= 0.01:
			base = [center[0], center[1], center_z + 0.01]
		else:
			base = [center[0], center[1], center_z - 0.02]

		utils.create_obj(p.GEOM_SPHERE,
		                 mass=-1,
		                 radius=0.008,
		                 rgbaColor=color,
		                 basePosition=base,
		                 baseOrientation=[0, 0, 0, 1]
		                 )


def add_forbidden_area(top_left, bottom_right):
	""" 1:1 pixel location"""

	length = bottom_right[0] - top_left[0]
	width = bottom_right[1] - top_left[1]

	center_x = (bottom_right[0] + top_left[0]) / 2
	center_y = (bottom_right[1] + top_left[1]) / 2

	utils.create_obj(p.GEOM_BOX,
	           mass=-1,
	           halfExtents=[length / 2, width / 2, 0.0001],
	           rgbaColor=[156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0, 1],
	           basePosition=[center_x, center_y, 0.01],
	           baseOrientation=[0, 0, 0, 1],
               object_list=forbidden_area
	           )

	top_left_pix = utils.xyz_to_pix([top_left[0], top_left[1], 0], bounds, 0.01)

	ob_list = []

	for i in range(int(length*100)):
		for j in range(int(width*100)):
			point = (top_left_pix[0] + j, 80 - (top_left_pix[1] + i))
			# point = (top_left_pix[1] + j, top_left_pix[0] + i)
			ob_list.append(point)

	analyzer.add_obstacles(ob_list)

	return [center_x-0.01, center_y-0.01]







arm = Jaco()


set_add_electrode()

# build_bin()
# add_obstacles()



for _ in range(100):
	p.stepSimulation()

# arm.pick_hold(object_list[0], [0.3, -0.2, 0.2], p.getQuaternionFromEuler([0,-np.pi, np.pi/2]))
# arm.place_hold(object_list[0], [0.5, -0.05, 0.07], p.getQuaternionFromEuler([0,-np.pi, 0]))

### depth map ###
depth = get_weight_map()
analyzer.set_map(depth)

# add_forbidden_area([0.3, -0.1], [0.35, 0.3])
# add_forbidden_area([0.45, -0.3], [0.55, 0.03])

# add_forbidden_area([0.3, -0.3], [0.35, 0.1])
# add_forbidden_area([0.6, -0.1], [0.65, 0.1])


analyzer.search()

success_1, path1, cost_1 = analyzer.get_result(0)
success_2, path2, cost_2 = analyzer.get_result(1)

show_path(depth, path2, 1)
show_path(depth, path1[:-40], 0)

# add_obstacles()
depth = get_weight_map()


path2 = []
for item in path1:
	path2.append([60 - item[0], item[1] + 10])

for i in range(10):
	path2.insert(0, [35, 15-i])

# show_path(depth, path2[:-10], 1)


# arm.pick_hold(object_list[0], [0.3, -0.2, 0.2], p.getQuaternionFromEuler([0,-np.pi, np.pi/2]))
# arm.place_hold(object_list[0], [0.6, 0, 0.07], p.getQuaternionFromEuler([0,-np.pi, np.pi/4]))



### save img ###
color, _, _ = render_camera(agent_cams[0])
rgb = color.copy()

rgb[:, :, 0] = rgb[:, :, 2]
rgb[:, :, 2] = color[:, :, 0]

# cv2.imwrite("/home/lima/Desktop/CircuitBot_RL/src/figures/2_2.png", rgb)
cv2.imwrite("4_2.png", rgb)






# analyzer.draw_map_3D()



for _ in range(10000):
	# obs = _get_obs()
	# img = get_image(obs)

	color, depth, msk = render_camera(agent_cams[0])
	# color, depth, msk = get_true_image()

	rgb = color.copy()

	rgb[:, :, 0] = rgb[:, :, 2]
	rgb[:,:,2] = color[:, :, 0]

	# cv2.imshow('test', img[:,:,3]*30)
	cv2.imshow('test', rgb)
	cv2.waitKey(1)
	p.stepSimulation()
	time.sleep(1/240)
