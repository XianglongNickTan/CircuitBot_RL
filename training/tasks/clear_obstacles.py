import collections
import numpy as np
from training.tasks.task import Task
# from ravens.utils import utils
import math
from training import utils
from gym import spaces
import cv2
from map_env.pathplanning.pathanalyzer import PathAnalyzer

import os, sys
import pybullet as p


rootdir = os.path.dirname(sys.modules['__main__'].__file__)
rootdir += "/obj_model"

obj_cube = rootdir + "/cube_4.obj"
obj_cuboid1 = rootdir + "/cuboid_4_4_8.obj"
obj_cuboid2 = rootdir + "/cuboid_4_16.obj"
obj_cuboid3 = rootdir + "/cuboid_8_8_4.obj"
obj_curve = rootdir + "/curve.obj"
obj_cylinder = rootdir + "/cylinder_4_4.obj"
obj_triangular_prism = rootdir + "/triangular_prism_4_8.obj"


class ClearObstaclesTask:
    """ remove one cube in the path"""

    def __init__(self,
                 env):

        self.mode = 'train'
        self.max_steps = 3


        self.objects = []

        self.weight_map = None

        # Workspace bounds.
        self.pix_size = 0.005
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])

        self.env = env
        self.arm = env.arm
        self.pixel_ratio = env.pixel_ratio

        self.action_space = spaces.Box(
            low=np.array([22 * self.pixel_ratio, 8 * self.pixel_ratio, 22 * self.pixel_ratio, 8 * self.pixel_ratio, 0]),
            high=np.array([58 * self.pixel_ratio, 47 * self.pixel_ratio, 58 * self.pixel_ratio, 47 * self.pixel_ratio, 1]),
            dtype=np.int)






    def init_task(self):
        object_1 = utils.create_obj(p.GEOM_MESH,
                                    mass=0.01,
                                    use_file=obj_cuboid2,
                                    rgbaColor=utils.COLORS['blue'],
                                    basePosition=[0.495, 0.075, 0.03],
                                    baseOrientation=p.getQuaternionFromEuler([0, 0, math.pi/2])
                                    )

        self.objects.append(object_1)




    def _update_weight_map(self):
        _, depth_map, _ = self.env.render_camera()

        x, y = depth_map.shape[0:2]

        #### resize the map to weight map ######
        weight_map = cv2.resize(depth_map, (int(y / self.pixel_ratio), int(x / self.pixel_ratio)))

        # weight_map = depth_map
        self.weight_map = weight_map



    def reset(self):
        self.init_task()

        self.analyzer = PathAnalyzer()
        self.path_length = 0
        self._update_weight_map()

        self.analyzer.set_map(self.weight_map)

        self.analyzer.set_pathplan(0, [self.env.ele_c_l, self.env.ele_r_n], [self.env.ele_c_l, self.env.ele_r_f])
        self.analyzer.set_pathplan(1, [self.env.ele_c_r, self.env.ele_r_n], [self.env.ele_c_r, self.env.ele_r_f])




    def reward(self):
        reward = 0

        self._update_weight_map()
        self.analyzer.set_map(self.weight_map)
        self.analyzer.search()

        success_1, path_1, cost_1 = self.analyzer.get_result(0)
        success_2, path_2, cost_2 = self.analyzer.get_result(1)

        if success_1:
            reward += 100
            reward -= cost_1

        if success_2:
            reward += 100
            reward -= cost_2

        return reward



def get_discrete_oracle_agent(self):
    OracleAgent = collections.namedtuple('OracleAgent', ['act'])



    def act(obs, info):  # pylint: disable=unused-argument
        """Calculate action."""

        raw_action = self.action_space.sample()
        place_xy = [raw_action[2], raw_action[3]]
        place_xy = utils.from_pixel_to_coordinate(place_xy, self._pixel_ratio)
        move_object = self.object[0]

        if raw_action[4] == 0:
            place_orin = [0, -math.pi, 0]
        else:
            place_orin = [0, -math.pi, math.pi / 2]


        place_background_height = self.weight_map[
            int(raw_action[2] / self.pixel_ratio), int(raw_action[3] / self.pixel_ratio)]


        base, pick_orin = p.getBasePositionAndOrientation(move_object)

        pick_orin = p.getEulerFromQuaternion(pick_orin)

        pick_z = base[2] + self._arm.grip_z_offset

        pick_orin = [0, -math.pi, pick_orin[2]]

        place_z = place_background_height / 100 + self._arm.grip_z_offset

        pick_point = [base[0], base[1], pick_z]
        place_point = [place_xy[0], place_xy[1], place_z]

        self.arm.pick_place_object(move_object, pick_point, pick_orin, place_point, place_orin)

        pick_orin_quat = p.getQuaternionFromEuler(pick_orin)
        pick_pose = ((base[0], base[1], pick_z),
                     (pick_orin_quat[0], pick_orin_quat[1], pick_orin_quat[2], pick_orin_quat[3]))
        pick_pose = (np.asarray(pick_pose[0]), np.asarray(pick_pose[1]))

        place_orin_quat = p.getQuaternionFromEuler(place_orin)
        place_pose = ((place_xy[0], place_xy[1], place_z),
                      (place_orin_quat[0], place_orin_quat[1], place_orin_quat[2], place_orin_quat[3]))
        place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))

        return {'pose0': pick_pose, 'pose1': place_pose}

    return OracleAgent(act)


