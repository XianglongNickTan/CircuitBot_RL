import pybullet as p
import pybullet_data
import os, sys
from jaco import Jaco
import time
import math


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




physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

client = PhysClientWrapper(p, physicsClient)

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")




arm = Jaco(p=client)

sphereRadius = 0.03

colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius])
mass = 1
visualShapeId = -1
basePosition = [0, 0.70, 0]
baseOrientation = [0, 0, 0, 1]

sphereUid = p.createMultiBody(mass, colBoxId, visualShapeId, basePosition,
                              baseOrientation)




ini_p = [0, 0.8, 0.05]
ori = [0, -math.pi, math.pi / 1.8]
end_p = [0, 0.7, 0.5]


# arm.pick_place_object(ini_p, ori, end_p, ori)

arm.move_to(ini_p, ori)
print("---------moved")

arm.create_gripper_constraints(sphereUid)

arm.pick()
print("---------picked")

arm.move_to(end_p, ori)
print("---------lefted")


arm.move_to(ini_p, ori)



arm.place()
print("---------placed")
arm.remove_gripper_constraints()


for _ in range(1000000):
    print("---------placed")

    print(arm.get_endEffector_pos())
    p.stepSimulation()
    time.sleep(1. / 240.)  # set time interval for visulaization

p.disconnect()