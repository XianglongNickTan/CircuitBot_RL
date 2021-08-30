import pybullet as p
import sys
import time
from collections import deque
import cv2
import numpy as np
import pybullet_data
from gym import spaces, Env
from gym.utils import seeding
import os, sys
import math
from pathplanning.map import Map
from pathplanning.analyzer import Analyzer


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



class MapEnv(Env):
    def __init__(self,
                 map_row=56,
                 map_column=40,
                 n_substeps=5,  # Number of simulation steps to do in every env step.
                 done_after=float("inf"),
                 use_gui=True,
                 workspace_width=56,
                 workspace_height=80,
                 y_offset=10
                 ):

        ### pybullet setting ###
        if use_gui:
            physics_client = p.connect(p.GUI)
        else:
            physics_client = p.connect(p.DIRECT)

        self.p = PhysClientWrapper(p, physics_client)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.y_offset = y_offset
        self.row = map_row
        self.column = map_column

        ### x - row - height
        ### y - colomn - width


        self.workspace_width = workspace_width
        self.workspace_height = workspace_height

        self.path_length = 0

        self.success = False


        self.light = {
            "diffuse": 0.4,
            "ambient": 0.5,
            "spec": 0.2,
            "dir": [10, 10, 100],
            "col": [1, 1, 1]}

        camera_center = (self.workspace_height / 2 + self.y_offset) / 100
        camera_height = 2

        self.viewMatrix = p.computeViewMatrix([camera_center, 0, camera_height], [camera_center, 0, -camera_height], [1, 0, 0])
        # self.viewMatrix = p.computeViewMatrix([0, camera_center, camera_height], [0, camera_center, -camera_height], [0, 1, 0])

        self.nearVal = 0.01
        self.farVal = camera_height

        fov = math.atan((self.workspace_height / 200) / camera_height)
        fov = fov * 180 / math.pi * 2
        print("-------------------")
        print(fov)

        self.projMatrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=1, nearVal=self.nearVal, farVal=self.farVal)

        self.objects = []


        ### initilize map ###
        self.init_sim()

        self.is_done = False

        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([self.row - 1, self.column - 1, self.row - 1, self.column - 1]),
            dtype=np.int)


        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.row, self.column), dtype='float32')



        self.pick_threshold = 0.005

        self.p.setGravity(0, 0, -10)


        self.analyzer = Analyzer()


    def _get_sim_image(self):
        pixel_ratio = 4
        width, height = self.workspace_height * pixel_ratio, self.workspace_height * pixel_ratio

        width_clip = int((self.workspace_height - self.workspace_width) * (pixel_ratio / 2))

        img = self.p.getCameraImage(
            width,
            height,
            self.viewMatrix,
            self.projMatrix,
            shadow=0)
            # lightAmbientCoeff=self.light["ambient"],
            # lightDiffuseCoeff=self.light["diffuse"],
            # lightSpecularCoeff=self.light["spec"],
            # lightDirection=self.light["dir"],
            # lightColor=self.light["col"])

        rgb = np.array(img[2], dtype=np.float).reshape(height, width, 4) / 255
        rgb[:, :, 3], rgb[:, :, 2] = rgb[:, :, 2], rgb[:, :, 0]
        rgb[:, :, 0] = rgb[:, :, 3]

        rgb_map = rgb[:, :, 0:3]
        rgb_map = rgb_map[:, width_clip:-width_clip, :]


        depth_map = np.array(img[3], dtype=np.float).reshape(height, width)
        depth_map = depth_map[:, width_clip:-width_clip]

        ### the distance from farVal(plane) to object height (m)####
        depth_map = self.farVal - self.farVal * self.nearVal / (self.farVal - (self.farVal - self.nearVal) * depth_map)

        rgb_d = np.dstack((rgb_map, depth_map))


        return rgb_map, depth_map, rgb_d


    def _get_weight_map(self):
        rgb_map, depth_map, _ = self._get_sim_image()

        depth_map *= 100
        depth_map -= 1

        x, y = depth_map.shape[0:2]

        #### resize the map to weight map ######
        weight_map = cv2.resize(depth_map, (int(y / 4), int(x / 4)))
        # weight_map = depth_map

        return weight_map



    def _create_obj(self, obj, mass=None, halfExtents=None, rgbaColor=None,
                   basePosition=None, baseOrientation=None, use_file=None):

        if not use_file:
            visual = self.p.createVisualShape(obj, halfExtents=halfExtents, rgbaColor=rgbaColor)
            shape = self.p.createCollisionShape(obj, halfExtents=halfExtents)

        else:
            visual = self.p.createVisualShape(obj, fileName=use_file, rgbaColor=rgbaColor)
            shape = self.p.createCollisionShape(obj, fileName=use_file)

        objID = self.p.createMultiBody(baseMass=mass,
                                  baseCollisionShapeIndex=shape,
                                  baseVisualShapeIndex=visual,
                                  basePosition=basePosition,
                                  baseOrientation=baseOrientation)

        return objID






    def init_sim(self):

        # planeId = self.p.loadURDF("plane.urdf")

        ##### create bottom plate ######
        # self._create_obj(self.p.GEOM_BOX,
        #                 mass=-1,
        #                 halfExtents=[self.workspace_width/200, self.workspace_height/200, 0.005],
        #                 rgbaColor=[1, 0, 1, 1],
        #                 basePosition=[0, self.workspace_height/200+0.1, 0.005],
        #                 baseOrientation=[0, 0, 0, 1]
        #                 )

        self._create_obj(self.p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[self.workspace_height/200, self.workspace_width/200, 0.005],
                        rgbaColor=[1, 0, 1, 1],
                        basePosition=[self.workspace_height/200+0.1, 0, 0.005],
                        baseOrientation=[0, 0, 0, 1]
                        )

        # ##### create wall for collision detect ######
        # wall1 = self._create_obj(self.p.GEOM_BOX,
        #                 mass=-1,
        #                 halfExtents=[0.001, 0.27, 0.2],
        #                 rgbaColor=[1, 1, 1, 1],
        #                 basePosition=[-0.2, 0.48, 1],
        #                 baseOrientation=[0, 0, 0, 1]
        #                 )
        #
        # wall2 = self._create_obj(self.p.GEOM_BOX,
        #                 mass=-1,
        #                 halfExtents=[0.2, 0.001, 0.2],
        #                 rgbaColor=[1, 1, 1, 1],
        #                 basePosition=[0, 0.76, 1],
        #                 baseOrientation=[0, 0, 0, 1]
        #                 )
        #
        # wall3 = self._create_obj(self.p.GEOM_BOX,
        #                 mass=-1,
        #                 halfExtents=[0.001, 0.27, 0.2],
        #                 rgbaColor=[1, 1, 1, 1],
        #                 basePosition=[0.2, 0.48, 1],
        #                 baseOrientation=[0, 0, 0, 1]
        #                 )
        #
        # wall4 = self._create_obj(self.p.GEOM_BOX,
        #                 mass=-1,
        #                 halfExtents=[0.2, 0.001, 0.2],
        #                 rgbaColor=[1, 1, 1, 1],
        #                 basePosition=[0, 0.20, 1],
        #                 baseOrientation=[0, 0, 0, 1]
        #                 )
        #
        # self.wall.append(wall1)
        # self.wall.append(wall2)
        # self.wall.append(wall3)
        # self.wall.append(wall4)

        object_1 = self._create_obj(self.p.GEOM_MESH,
                        mass=0.01,
                        use_file=obj_cuboid2,
                        rgbaColor=[1, 1, 1, 1],
                        basePosition=[0.5, 0.08, 0.05],
                        baseOrientation=self.p.getQuaternionFromEuler([0,0,math.pi/2])
                        )


        object_1 = self._create_obj(self.p.GEOM_MESH,
                        mass=0.01,
                        use_file=obj_cuboid2,
                        rgbaColor=[1, 1, 1, 1],
                        basePosition=[0.5, -0.08, 0.05],
                        baseOrientation=self.p.getQuaternionFromEuler([0,0,math.pi/2])
                        )

        object_1 = self._create_obj(self.p.GEOM_MESH,
                        mass=0.01,
                        use_file=obj_triangular_prism,
                        rgbaColor=[1, 1, 1, 1],
                        basePosition=[0.7, 0, 0.05],
                        baseOrientation=self.p.getQuaternionFromEuler([0,0,math.pi/2])
                        )

        self.objects.append(object_1)


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



    def cal_show_path(self):
        weight_map = self._get_weight_map()
        self.analyzer.updateMap(weight_map)

        self.analyzer.update_pole_pair(0,[42,5],[36,40])
        self.analyzer.update_pole_pair(1,[25,5],[25,45])

        self.analyzer.search()
        self.analyzer.draw_map_3D()

    def show_map(self):
        # rgb_map, depth_map, _ = self.get_sim_image()

        rgb,_,_ = self._get_sim_image()
        cv2.imshow("test", rgb)
        cv2.waitKey(1)


    def _action_from_pixel_to_coordinate(self, action):
        pass


    def _apply_action(self, action):
        """ apply action to update the map.
            action = array([x1, y1, x2, y2])
            x height
            y width
        """

        # action = np.clip(action, [7, 24, 7, 24], [48, 72, 48, 72])


        pick_point = [action[0], action[1]]
        place_point = [action[2], action[3]]

        move_object = self._compare_object_base(pick_point)

        #### orin #####
        # orin = action[4]


        if move_object:
            base, orin = self.p.getBasePositionAndOrientation(move_object)





            ###### z value needs to be edited based on height map ########
            place_point = [action[2], action[3], 0.1]

            self.p.resetBasePositionAndOrientation(move_object,
                                                   place_point,
                                                   orin)



        return self.is_done



    def _compare_object_base(self, pick_pos):
        move_object = None

        max_z = 0

        for object in self.objects:
            base, orin = self.p.getBasePositionAndOrientation(object)
            object_z = base[2]

            if pick_pos[0] - self.pick_threshold < base[0] < pick_pos[0] + self.pick_threshold:
                if pick_pos[1] - self.pick_threshold < base[1] < pick_pos[1] + self.pick_threshold:
                    if object_z > max_z:
                        max_z = object_z
                        move_object = object

            else:
                continue

        return move_object









    def get_obs(self):
        rgb_map, depth_map, rgb_d = self._get_sim_image()

        return obs




    def step(self, action):
        """ Execute one time step within the environment."""
        pass


    def render(self, mode="human"):
        pass



###### test map class #####
my_map = MapEnv()

pick_y = p.addUserDebugParameter("pick",0,1,0)
place_y = p.addUserDebugParameter("place",0,1,0)


# add_test_obj()

for i in range(10):

    y_1 = p.readUserDebugParameter(pick_y)
    y_2 = p.readUserDebugParameter(place_y)

    action = [0,y_1, 0, y_2]
    my_map._apply_action(action)

    p.stepSimulation()
    time.sleep(1./240.)
    my_map.show_map()

my_map.cal_show_path()

