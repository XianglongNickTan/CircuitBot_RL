import os, sys

from absl import app
from absl import flags
import cv2
import pybullet as p
import random
import numpy as np

from ravens.dataset import Dataset

from env.environment_old import Environment
# from tasks.clear_obstacles import ClearObstaclesTask
from tasks.clear_obstacles import ClearObstaclesTask
from utils.pathplanning.pathanalyzer import PathAnalyzer
from utils import utils
from tasks.cameras import DaBai



flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_bool('disp', True, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_integer('n', 100, '')
flags.DEFINE_integer('steps_per_seg', 3, '')
flags.DEFINE_string('task', 'clear_one_obstacle', '')

FLAGS = flags.FLAGS


def main(unused_argv):

	env = Environment(
		disp=FLAGS.disp,
		shared_memory=FLAGS.shared_memory,
		hz=240)

	pix_size = 0.0025


	analyzer = PathAnalyzer()
	bounds = np.array([[0.1, 0.9], [-0.28, 0.28], [0, 0.3]])

	env.reset()

	objects = []
	print(utils.OBJECTS['cuboid2'])

	utils.create_obj(p.GEOM_MESH,
	                 mass=0.01,
	                 use_file=utils.OBJECTS['cuboid2'],
	                 rgbaColor=utils.COLORS['red'],
	                 basePosition=[0.325,
	                               0.15 * (2 * random.random() - 1), 0.03],
	                 baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]),
	                 object_list=objects
	                 )
	#
	utils.create_obj(p.GEOM_MESH,
	                 mass=0.01,
	                 use_file=utils.OBJECTS['cuboid2'],
	                 rgbaColor=utils.COLORS['red'],
	                 basePosition=[0.475,
	                               0.15 * (2 * random.random() - 1), 0.03],
	                 baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]),
	                 object_list=objects
	                 )
	#
	utils.create_obj(p.GEOM_MESH,
	                 mass=0.01,
	                 use_file=utils.OBJECTS['cuboid2'],
	                 rgbaColor=utils.COLORS['red'],
	                 basePosition=[0.625,
	                               0.15 * (2 * random.random() - 1), 0.03],
	                 baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]),
	                 object_list=objects
	                 )

	for _ in range(500):
		p.stepSimulation()

	_, depth_map, obj_mask = get_true_image(env)
	# _, depth_map= env._get_obs()

	print("--------")

	x, y = depth_map.shape[0:2]

	print(x, y)


	# depth_map *= 100  ### convert to cm
	# depth_map -= 1  ### minus plate height

	#### resize the map to weight map ######
	weight_map = cv2.resize(depth_map, (int(y / 0.01 * pix_size), int(x / 0.01 * pix_size)))


	analyzer.set_map(weight_map)

	print(weight_map.shape)
	# analyzer.set_map(np.ones([80, 64]))

	# analyzer.set_pathplan(0, [robot_column, robot_row], [source_column, source_row])
	# analyzer.set_pathplan(1, [55 - robot_column, robot_row], [55 - source_column, source_row])
	
	# print([robot_column, robot_row], [source_column, source_row])
	# print([55 - robot_column, robot_row], [55 - source_column, source_row])


	#
	#
	# analyzer.set_pathplan(0, [22, 65], [22, 20])
	# analyzer.set_pathplan(1, [33, 65], [33, 20])
	#
	# analyzer.set_pathplan(0, [35, 79], [35, 0])
	# analyzer.set_pathplan(1, [25, 79], [25, 0])

	analyzer.set_pathplan(0, [0, 20], [70, 20])
	analyzer.set_pathplan(1, [0, 36], [70, 36])

	analyzer.search()
	analyzer.draw_map_3D()

camera = DaBai.CONFIG

bounds = np.array([[0.1, 0.9], [-0.28, 0.28], [0, 0.3]])

pix_size = 0.0025


def get_true_image(env):
	"""Get RGB-D orthographic heightmaps and segmentation masks."""

	# Capture near-orthographic RGB-D images and segmentation masks.
	color, depth, segm = env.render_camera(camera[0])

	# Combine color with masks for faster processing.
	color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

	# Reconstruct real orthographic projection from point clouds.
	hmaps, cmaps = utils.reconstruct_heightmaps(
		[color], [depth], camera, bounds, pix_size)

	# Split color back into color and masks.
	cmap = np.uint8(cmaps)[0, Ellipsis, :3]
	hmap = np.float32(hmaps)[0, Ellipsis]
	mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
	return cmap, hmap, mask


if __name__ == '__main__':
	app.run(main)
