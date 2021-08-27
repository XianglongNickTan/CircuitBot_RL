import numpy as np
from gym import spaces, Env
import random
import os, sys
import math

### delete after test###
import pybullet as p
import cv2
import time



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


#######################################################



class Map():
    def __init__(self,
                 p,
                 randomize_obstacle=False,  # Whether obstacles initial position should be randomized.
                 map_row=8,
                 map_column=8):

        self.p = p

        self.row = map_row
        self.column = map_column

        self.weight_map = np.zeros((self.row, self.column), dtype="float32")

        self.pre_map = self.weight_map

        self.path_length = 0

        self.success = False

        self.shape_rec = np.ones((1, 3), dtype="float32")

        self.init_sim()

        self.light = {
            "diffuse": 0.4,
            "ambient": 0.5,
            "spec": 0.2,
            "dir": [10, 10, 100],
            "col": [1, 1, 1]}

        self.viewMatrix = p.computeViewMatrix([0, 0.47, 1], [0, 0.47, -1], [0, 1, 0])

        self.nearVal = 0.01
        self.farVal = 1

        self.projMatrix = p.computeProjectionMatrixFOV(
            fov=30, aspect=1, nearVal=self.nearVal, farVal=self.farVal)


    def get_sim_image(self):
        pixel_ratio = 4
        width, height = 54 * pixel_ratio, 54 * pixel_ratio

        width_clip = int(14 * (pixel_ratio / 2))

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


    def get_weight_map(self):
        rgb_map, depth_map, _ = self.get_sim_image()

        depth_map *= 100
        depth_map -= 1

        x, y = depth_map.shape[0:2]

        #### resize the map to weight map ######
        weight_map = cv2.resize(depth_map, (int(y / 4), int(x / 4)))
        # weight_map = depth_map



        return weight_map



    def show_map(self):
        # rgb_map, depth_map, _ = self.get_sim_image()

        weight_map = self.get_weight_map()
        cv2.imshow("test", weight_map)
        cv2.waitKey(1)



    def create_obj(self, obj, mass=None, halfExtents=None, rgbaColor=None,
                   basePosition=None, baseOrientation=None, use_file=None):

        if not use_file:
            visual = p.createVisualShape(obj, halfExtents=halfExtents, rgbaColor=rgbaColor)

            shape = p.createCollisionShape(obj, halfExtents=halfExtents)

        else:
            visual = p.createVisualShape(obj, fileName=use_file, rgbaColor=rgbaColor)

            shape = p.createCollisionShape(obj, fileName=use_file)


        objID = p.createMultiBody(baseMass=mass,
                                  baseCollisionShapeIndex=shape,
                                  baseVisualShapeIndex=visual,
                                  basePosition=basePosition,
                                  baseOrientation=baseOrientation)

        return objID


    def init_sim(self):
        """ init simulation env."""

        ##### create bottom plate ######
        plateID = self.create_obj(p.GEOM_BOX,
                        mass=-1,
                        halfExtents=[0.2, 0.27, 0.005],
                        rgbaColor=[1, 1, 1, 1],
                        basePosition=[0, 0.47, 0.005],
                        baseOrientation=[0, 0, 0, 1]
                        )


        ##### create random object ######
        # boxID = self.create_obj(p.GEOM_BOX,
        #                 mass=1,
        #                 halfExtents=[0.08, 0.02, 0.03],
        #                 rgbaColor=[0, 1, 1, 1],
        #                 basePosition=[0, 0.47, 0.03],
        #                 baseOrientation=[0, 0, 0, 1]
        #                 )

        boxID = self.create_obj(p.GEOM_MESH,
                        mass=1,
                        basePosition=[0, 0.47, 0.03],
                        rgbaColor=[0, 1, 1, 1],
                        use_file=obj_cube
                        )

        boxID = self.create_obj(p.GEOM_MESH,
                        mass=1,
                        basePosition=[0, 0.55, 0.035],
                        # baseOrientation= self.p.getQuaternionFromEuler([0,1.315,0]),
                        rgbaColor=[0, 1, 1, 1],
                        use_file=obj_triangular_prism
                        )

        boxID = self.create_obj(p.GEOM_MESH,
                        mass=1,
                        basePosition=[0, 0.42, 0.03],
                        baseOrientation= self.p.getQuaternionFromEuler([0,0,0]),
                        rgbaColor=[0, 0, 0, 1],
                        use_file=obj_curve
                        )

        boxID = self.create_obj(p.GEOM_MESH,
                        mass=1,
                        basePosition=[-0.1, 0.47, 0.03],
                        # baseOrientation=self.p.getQuaternionFromEuler([1/2*math.pi, 0, 0]),
                        rgbaColor=[0, 1, 1, 1],
                        use_file=obj_cuboid1
                        )

        boxID = self.create_obj(p.GEOM_MESH,
                        mass=1,
                        basePosition=[0.1, 0.47, 0.03],
                        rgbaColor=[0, 1, 1, 1],
                        use_file=obj_cuboid2
                        )


        boxID = self.create_obj(p.GEOM_MESH,
                        mass=1,
                        basePosition=[0.15, 0.47, 0.03],
                        rgbaColor=[0, 1, 1, 1],
                        use_file=obj_cylinder
                        )

        boxID = self.create_obj(p.GEOM_MESH,
                        mass=1,
                        basePosition=[-0.15, 0.47, 0.03],
                        rgbaColor=[0, 1, 1, 1],
                        use_file=obj_cuboid3
                        )

    def apply_action(self, action):
        """ apply action to update the map.
            action = [x, y]
        """
        move_point = action

        action = np.clip(action, [0, 1], [7, 6])

        self.weight_map = np.zeros((self.row, self.column), dtype="float32")

        self.weight_map[action[0], action[1] - 1: action[1] + 2] = self.shape_rec




###### test map class #####
physics_client = p.connect(p.GUI)
p = PhysClientWrapper(p, physics_client)
my_map = Map(p)
p.setGravity(0,0,-10)


# add_test_obj()

for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)
    my_map.show_map()


