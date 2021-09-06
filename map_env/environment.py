import pybullet as p
import time
from collections import deque
import cv2
import numpy as np
import pybullet_data
from gym import spaces, Env
from gym.utils import seeding
import os, sys
import math
from map_env.pathplanning.pathanalyzer import PathAnalyzer
import pandas as pd

from robot_env.jaco import Jaco


sys.path.insert(1, "../bullet3/build_cmake/examples/pybullet")
timeStep = 1 / 240.0

rootdir = os.path.dirname(sys.modules['__main__'].__file__)
rootdir += "/obj_model"

obj_cube = rootdir + "/cube_4.obj"
obj_cuboid1 = rootdir + "/cuboid_4_4_8.obj"
obj_cuboid2 = rootdir + "/cuboid_4_16.obj"
obj_cuboid3 = rootdir + "/cuboid_8_8_4.obj"
obj_curve = rootdir + "/curve.obj"
obj_cylinder = rootdir + "/cylinder_4_4.obj"
obj_triangular_prism = rootdir + "/triangular_prism_4_8.obj"

np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

from ravens.environments.environment import Environment as Envi

class Environment(Envi):
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
                 use_egl=False
                 ):

        ### pybullet setting ###
        if disp:
            physics_client = p.connect(p.GUI)
        else:
            physics_client = p.connect(p.DIRECT)

        self.p = PhysClientWrapper(p, physics_client)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, 0)
        self.p.setGravity(0, 0, -10)

        ### map settings ###
        ### x - row - height
        ### y - colomn - width
        self.plate_offset = plate_offset
        self.row = map_row
        self.column = map_column

        self.workspace_width = workspace_width
        self.workspace_height = workspace_height


        ### robot pick place settings ###
        self.pick_threshold = 0.025     ## m
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

        self.ele_n_l = self._from_pixel_to_coordinate([self.ele_r_n, self.ele_c_l], self.weight_map_ratio)
        self.ele_n_r = self._from_pixel_to_coordinate([self.ele_r_n, self.ele_c_r], self.weight_map_ratio)
        self.ele_f_l = self._from_pixel_to_coordinate([self.ele_r_f, self.ele_c_l], self.weight_map_ratio)
        self.ele_f_r = self._from_pixel_to_coordinate([self.ele_r_f, self.ele_c_r], self.weight_map_ratio)



        ### action = [x, y, x, y, ori] ###
        self.action_space = spaces.Box(
            low=np.array([22*self.pixel_ratio, 8*self.pixel_ratio, 22*self.pixel_ratio, 8*self.pixel_ratio, 0]) ,
            high=np.array([58*self.pixel_ratio, 47*self.pixel_ratio, 58*self.pixel_ratio, 47*self.pixel_ratio, 1]),
            dtype=np.int)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.row, self.column), dtype='float32')


        ### sim camera settings ###
        self.light = {
            "diffuse": 0.4,
            "ambient": 0.5,
            "spec": 0.2,
            "dir": [10, 10, 100],
            "col": [1, 1, 1]}

        camera_center = (self.workspace_height / 2 + self.plate_offset) / 100
        camera_height = 2
        self.viewMatrix = p.computeViewMatrix([camera_center, 0, camera_height], [camera_center, 0, -camera_height], [1, 0, 0])
        self.nearVal = 0.01
        self.farVal = camera_height
        fov = math.atan((self.workspace_height / 200) / camera_height)
        fov = fov * 180 / math.pi * 2
        self.projMatrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=1, nearVal=self.nearVal, farVal=self.farVal)



    def _get_sim_image(self):
        width, height = self.workspace_height * self.pixel_ratio, self.workspace_height * self.pixel_ratio

        width_clip = int((self.workspace_height - self.workspace_width) * (self.pixel_ratio / 2))

        img = self.p.getCameraImage(
            width,
            height,
            self.viewMatrix,
            self.projMatrix,
            shadow=0)

        rgb = np.array(img[2], dtype=np.float).reshape(height, width, 4) / 255
        rgb[:, :, 3], rgb[:, :, 2] = rgb[:, :, 2], rgb[:, :, 0]
        rgb[:, :, 0] = rgb[:, :, 3]

        rgb_map = rgb[:, :, 0:3]
        rgb_map = rgb_map[:, width_clip:-width_clip, :]

        depth_map = np.array(img[3], dtype=np.float).reshape(height, width)
        depth_map = depth_map[:, width_clip:-width_clip]

        ### the distance from farVal(plane) to object height (m)####
        depth_map = self.farVal - self.farVal * self.nearVal / (self.farVal - (self.farVal - self.nearVal) * depth_map)
        depth_map *= 100    ### convert to cm
        depth_map -= 1      ### minus plate height

        rgb_d = np.dstack((rgb_map, depth_map))

        return rgb_map, depth_map, rgb_d


    def _update_weight_map(self):
        _, depth_map, _ = self._get_sim_image()

        x, y = depth_map.shape[0:2]

        #### resize the map to weight map ######
        weight_map = cv2.resize(depth_map, (int(y / self.pixel_ratio), int(x / self.pixel_ratio)))

        # weight_map = depth_map
        self.weight_map = weight_map


    def _create_obj(self, obj, mass=None, halfExtents=None, radius=None, height=None, rgbaColor=None,
                   basePosition=None, baseOrientation=None, use_file=None):

        if not use_file:

            if obj == self.p.GEOM_BOX:
                visual = self.p.createVisualShape(obj, halfExtents=halfExtents, rgbaColor=rgbaColor)
                shape = self.p.createCollisionShape(obj, halfExtents=halfExtents)

            elif obj == self.p.GEOM_CYLINDER:
                # visual = self.p.createVisualShape(obj, radius=radius, length=height, rgbaColor=rgbaColor)
                visual = self.p.createVisualShape(obj, radius=radius, length=height, rgbaColor=rgbaColor)
                # shape = self.p.createCollisionShape(obj, radius=radius, height=height)
                shape = -1

            else:
                raise NotImplementedError()

        else:
            visual = self.p.createVisualShape(obj, fileName=use_file, rgbaColor=rgbaColor)
            shape = self.p.createCollisionShape(obj, fileName=use_file)

        objID = self.p.createMultiBody(baseMass=mass,
                                  baseCollisionShapeIndex=shape,
                                  baseVisualShapeIndex=visual,
                                  basePosition=basePosition,
                                  baseOrientation=baseOrientation)

        return objID


    def _init_sim(self):

        self.p.loadURDF("plane.urdf")

        ### create plate ###
        self._create_obj(self.p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[self.workspace_height/200, self.workspace_width/200, 0.005],
                        rgbaColor=[1, 0.90, 0.72, 1],
                        basePosition=[self.workspace_height/200+0.1, 0, 0.005],
                        baseOrientation=[0, 0, 0, 1]
                        )

        ### create near electrode ###
        self._create_obj(self.p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[0.0075, 0.0075, 0.0001],
                        rgbaColor=[0, 0, 0, 1],
                        basePosition=[self.ele_n_l[0], self.ele_n_l[1], 0.01],
                        baseOrientation=[0, 0, 0, 1]
                        )

        self._create_obj(self.p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[self.electrode_x_offset / 200, 0.005, 0.0001],
                        rgbaColor=[0, 0, 0, 1],
                        basePosition=[(self.ele_n_l[0] + 0.1) / 2, self.ele_n_l[1], 0.01],
                        baseOrientation=[0, 0, 0, 1]
                        )

        self._create_obj(self.p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[0.0075, 0.0075, 0.0001],
                        rgbaColor=[1, 1, 1, 1],
                        basePosition=[self.ele_n_r[0], self.ele_n_r[1], 0.01],
                        baseOrientation=[0, 0, 0, 1]
                        )

        self._create_obj(self.p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[self.electrode_x_offset / 200, 0.005, 0.0001],
                        rgbaColor=[1, 1, 1, 1],
                        basePosition=[(self.ele_n_r[0] + 0.1) / 2, self.ele_n_r[1], 0.01],
                        baseOrientation=[0, 0, 0, 1]
                        )


        ### create far electrode ###
        self._create_obj(self.p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[0.0075, 0.0075, 0.0001],
                        rgbaColor=[0, 0, 0, 1],
                        basePosition=[self.ele_f_l[0], self.ele_f_l[1], 0.01],
                        baseOrientation=[0, 0, 0, 1]
                        )


        self._create_obj(self.p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[self.electrode_x_offset / 200, 0.005, 0.0001],
                        rgbaColor=[0, 0, 0, 1],
                        basePosition=[self.ele_f_l[0] + self.electrode_x_offset / 200, self.ele_f_l[1], 0.01],
                        baseOrientation=[0, 0, 0, 1]
                        )

        self._create_obj(self.p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[0.0075, 0.0075, 0.0001],
                        rgbaColor=[1, 1, 1, 1],
                        basePosition=[self.ele_f_r[0], self.ele_f_r[1], 0.01],
                        baseOrientation=[0, 0, 0, 1]
                        )


        self._create_obj(self.p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[self.electrode_x_offset / 200, 0.005, 0.0001],
                        rgbaColor=[1, 1, 1, 1],
                        basePosition=[self.ele_f_r[0] + self.electrode_x_offset / 200, self.ele_f_r[1], 0.01],
                        baseOrientation=[0, 0, 0, 1]
                        )





    def _check_if_out_workspace(self, object, wall):
        P_min, P_max = self.p.getAABB(object)
        id_tuple = self.p.getOverlappingObjects(P_min, P_max)

        if len(id_tuple) > 1:
            for ID, _ in id_tuple:
                if ID == wall:
                    return True

                else:
                    continue

        return False


    def _show_3d_weightmap_path(self):
        self.analyzer.draw_map_3D()


    def _from_pixel_to_coordinate(self, x_y, ratio):
        """ ratio: 1 = 1:1 cm  2 = 1:0.5cm """
        real_x = self.workspace_height - (0.5 + x_y[0]) / ratio
        real_y = self.workspace_width / 2 - (0.5 + x_y[1]) / ratio

        return [(real_x + self.plate_offset) / 100, real_y / 100]



    def _add_obstacles(self, top_left, bottom_right):

        length = bottom_right[0] - top_left[0] + 1
        width = bottom_right[1] - top_left[1] + 1

        center_x = (bottom_right[0] + top_left[0]) / 2
        center_y = (bottom_right[1] + top_left[1]) / 2

        center = self._from_pixel_to_coordinate([center_x, center_y], self.weight_map_ratio)

        ob_list = []

        for i in range(length):
            for j in range(width):
                point = (top_left[1] + j, top_left[0] + i)
                ob_list.append(point)

        self.analyzer.set_obstacles(ob_list)

        self._create_obj(self.p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[length/200, width/200, 0.0001],
                        rgbaColor=[1, 0, 0, 1],
                        basePosition=[center[0], center[1], 0.01],
                        baseOrientation=[0, 0, 0, 1]
                        )


    def _show_path(self, path):
        for point in path:
            center_x = point[1]
            center_y = point[0]

            center_x = self.workspace_height - center_x - 1
            center_z = self.weight_map[center_x, center_y] / 100

            center = self._from_pixel_to_coordinate([center_x, center_y], self.weight_map_ratio)

            self._create_obj(self.p.GEOM_BOX,
                             mass=-1,
                             halfExtents=[0.005, 0.005, 0.0001],
                             rgbaColor=[0, 0, 0, 1],
                             basePosition=[center[0], center[1], center_z + 0.01],
                             baseOrientation=[0, 0, 0, 1]
                             )

    def _add_object(self):
        self.objects = []

        object_1 = self._create_obj(self.p.GEOM_MESH,
                        mass=0.01,
                        use_file=obj_cuboid2,
                        rgbaColor=[1, 0, 1, 1],
                        basePosition=[0.495, 0.075, 0.03],
                        baseOrientation=self.p.getQuaternionFromEuler([0,0,math.pi/2])
                        )

        object_2 = self._create_obj(self.p.GEOM_MESH,
                        mass=0.01,
                        use_file=obj_cuboid2,
                        rgbaColor=[1, 0, 1, 1],
                        basePosition=[0.6,-0.1, 0.03],
                        baseOrientation=self.p.getQuaternionFromEuler([0,0,math.pi/2])
                        )

        object_3 = self._create_obj(self.p.GEOM_MESH,
                        mass=0.01,
                        use_file=obj_cuboid2,
                        rgbaColor=[1, 0, 1, 1],
                        basePosition=[0.7, 0, 0.03],
                        baseOrientation=self.p.getQuaternionFromEuler([0,0,math.pi/2])
                        )


        self.objects.append(object_1)
        self.objects.append(object_2)
        self.objects.append(object_3)

    def _compare_object_base(self, pick_pos):
        move_object = None
        max_z = 0

        for object in self.objects:
            base, orin = self.p.getBasePositionAndOrientation(object)
            object_z = base[2]

            if pick_pos[0] - self.pick_threshold <= base[0] <= pick_pos[0] + self.pick_threshold:
                if pick_pos[1] - self.pick_threshold <= base[1] <= pick_pos[1] + self.pick_threshold:
                    if object_z > max_z:
                        max_z = object_z
                        move_object = object

            else:
                continue

        return move_object


    def _apply_action(self, raw_action):
        """ apply action to update the map."""

        pick_xy = [raw_action[0], raw_action[1]]
        place_xy = [raw_action[2], raw_action[3]]

        pick_xy = self._from_pixel_to_coordinate(pick_xy, self.pixel_ratio)
        place_xy = self._from_pixel_to_coordinate(place_xy, self.pixel_ratio)

        if raw_action[4] == 0:
            place_orin = [0, -math.pi, 0]
        else:
            place_orin = [0, -math.pi, math.pi / 2]

        pick_background_height = self.weight_map[int(raw_action[0] / self.pixel_ratio), int(raw_action[1] / self.pixel_ratio)]
        place_background_height = self.weight_map[int(raw_action[2] / self.pixel_ratio), int(raw_action[3] / self.pixel_ratio)]

        move_object = self._compare_object_base(pick_xy)

        if move_object:
            base, pick_orin = self.p.getBasePositionAndOrientation(move_object)
            pick_orin = self.p.getEulerFromQuaternion(pick_orin)
            pick_z = base[2] + self.arm.grip_z_offset
            pick_orin = [0, -math.pi, pick_orin[2] + self.arm.ori_offset]

        else:
            pick_z = pick_background_height / 100 + self.arm.grip_z_offset
            pick_orin = self.arm.restOrientation

        place_z = place_background_height / 100 + self.arm.grip_z_offset

        pick_point = [pick_xy[0], pick_xy[1], pick_z]
        place_point = [place_xy[0], place_xy[1], place_z]

        self.arm.pick_place_object(move_object, pick_point, pick_orin, place_point, place_orin)



    def _get_obs(self):
        _, _, rgb_d = self._get_sim_image()

        return rgb_d


    def _get_reward(self):
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

        # self._show_path(path_1)
        # self._show_path(path_2)

        return reward


    def _is_success(self, reward):
        pass


    def reset(self):
        ### init map ###
        self._init_sim()
        self._add_object()

        self.weight_map = None

        ### init jaco ###
        self.arm = Jaco(self.p)

        ### init path planning ###
        self.analyzer = PathAnalyzer()
        self.path_length = 0
        self._update_weight_map()
        self.analyzer.set_map(self.weight_map)
        self.analyzer.set_pathplan(0,[self.ele_c_l,self.ele_r_n],[self.ele_c_l,self.ele_r_f])
        self.analyzer.set_pathplan(1,[self.ele_c_r,self.ele_r_n],[self.ele_c_r,self.ele_r_f])

        for _ in range(1000):             ### stablize init
            self.p.stepSimulation()

        init_obs = self._get_obs()
        self._update_weight_map()

        return init_obs


    def step(self, action):
        """ Execute one time step within the environment."""

        if self.numSteps == 0:
            self.startTime = time.time()

        self._apply_action(action)

        # for i in range(self.n_substeps):
        #     self.p.stepSimulation()

        self.numSteps += 1

        current_obs = self._get_obs()

        info = {}
        reward = self._get_reward()
        done = self._is_success(reward) or self.numSteps > self.doneAfter
        if done:
            info = {"episode": {"l": self.numSteps, "r": reward}}

        return current_obs, reward, done, info


    def render(self, mode="human"):
        rgb, _, _ = self._get_sim_image()
        if mode == "rgb_array":
            return rgb

        elif mode == "human":
            cv2.imshow("test", rgb)
            cv2.waitKey(1)


class PhysClientWrapper:
    """
    This is used to make sure each BulletRobotEnv has its own physicsClient and
    they do not cross-communicate.
    """

    def __init__(self, other, physics_client_id):
        self.other = other
        self.physicsClientId = physics_client_id

    def __getattr__(self, name):
        if hasattr(self.other, name):
            attr = getattr(self.other, name)
            if callable(attr):
                return lambda *args, **kwargs: self._wrap(attr, args, kwargs)
            return attr
        raise AttributeError(name)

    def _wrap(self, func, args, kwargs):
        kwargs["physicsClientId"] = self.physicsClientId
        return func(*args, **kwargs)
