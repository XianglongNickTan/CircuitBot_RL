import pybullet as p
import time
import cv2
import pybullet_data
import numpy as np
import os, sys
from robot_env.jaco import Jaco

# rootdir = os.path.dirname(sys.modules['__main__'].__file__)
# urdf = rootdir + "/../robot_env/jaco_description/urdf/j2n6s300_twofingers.urdf"
#
# print(rootdir)
#
# print(os.path.dirname(os.path.abspath("__file__")))
# print(os.path.pardir)

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
# planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])


# trayId = p.loadURDF("table_square/table_square.urdf")

# robot = p.loadURDF(urdf,
#                    useFixedBase=1)


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




# physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

# client = PhysClientWrapper(p, physicsClient)

# arm = Jaco(p=client)

mass = -1
# visualShapeId = -1


visualShapeId = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[0.2, 0.27, 0.005],
                                    rgbaColor=[0,1,1,1])

colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[0.2, 0.27, 0.005])

basePosition = [0, 0.47, 0.005]
baseOrientation = [0,0,0,1]

sphereUid = p.createMultiBody(mass,
                              colBoxId,
                              visualShapeId,
                              basePosition,
                              baseOrientation)



viewMatrix = p.computeViewMatrix([0, 0.47, 2], [0, 0.47, -1],
                                      [0, 1, 0])

projMatrix = p.computeProjectionMatrixFOV(
    fov=15.37, aspect=1, nearVal=0.01, farVal=2)

# projMatrix = p.computeProjectionMatrix(
#     left=-0.2, right=0.2, top=0.74, bottom=0.2, nearVal=0.01, farVal=2.5
# )



sphereRadius = 0.02

colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius])

colBoxId2 = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[0.08, sphereRadius, sphereRadius])

mass = -1
visualShapeId = -1
basePosition = [0, 0.50, 0.06]
baseOrientation = [0, 0, 0, 1]

visualShapeId2 = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[sphereRadius, sphereRadius, sphereRadius],
                                    rgbaColor=[1,0,1,1])


boxUid = p.createMultiBody(mass, colBoxId, visualShapeId2, basePosition,
                              baseOrientation)
basePosition = [0.05, 0.3, 0.06]

visualShapeId2 = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[0.08, sphereRadius, sphereRadius],
                                    rgbaColor=[1,1,0,1])


box2= p.createMultiBody(mass, colBoxId2, visualShapeId2, basePosition,
                              baseOrientation)

print(p.getBasePositionAndOrientation(box2))

light = {
    "diffuse": 0.4,
    "ambient": 0.5,
    "spec": 0.2,
    "dir": [10, 10, 100],
    "col": [1, 1, 1]}

def render(mode='human'):
    width, height = 108, 108

    pixel_ratio = 2
    width, height = 54 * pixel_ratio, 54 * pixel_ratio

    width_clip = int(14 * (pixel_ratio / 2))

    img = p.getCameraImage(
        width,
        height,
        viewMatrix,
        projMatrix,
        shadow=0
        # lightAmbientCoeff=light["ambient"],
        # lightDiffuseCoeff=light["diffuse"],
        # lightSpecularCoeff=light["spec"],
        # lightDirection=light["dir"],
        # lightColor=light["col"]
    )

    # img = p.getCameraImage(
    #     width,
    #     height
    # )

    # rgb = rgb

    rgb = np.array(img[2], dtype=np.float).reshape(height, width, 4) / 255
    rgb[:, :, 3], rgb[:, :, 2] = rgb[:, :, 2], rgb[:, :, 0]
    rgb[:, :, 0] = rgb[:, :, 3]

    rgb_map = rgb[:, :, 0:3]
    rgb_map = rgb_map[:, width_clip:-width_clip, :]

    depth_map = np.array(img[3], dtype=np.float).reshape(height, width)
    depth_map = depth_map[:, width_clip:-width_clip]

    rgb_d = np.dstack((rgb_map, depth_map))


    if mode == 'rgb_array':
        return rgb_map


    elif mode == 'rgb_depth_array':
        return rgb_d

    elif mode == 'human':
        # cv2.imshow("test", rgb_map)
        cv2.imshow("test", rgb_map)
        cv2.waitKey(1)

# width, height = 106, 84

# render()

# w, h, rgba, depth, mask = p.getCameraImage(
#     width=width,
#     height=height,
#     projectionMatrix=projMatrix,
#     viewMatrix=viewMatrix,
#     shadow=1,
#     lightAmbientCoeff=light["ambient"],
#     lightDiffuseCoeff=light["diffuse"],
#     lightSpecularCoeff=light["spec"],
#     lightDirection=light["dir"],
#     lightColor=light["col"]
# )





for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
    render()

# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)





p.disconnect()