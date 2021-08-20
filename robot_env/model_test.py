import pybullet as p
import os, sys



rootdir = os.path.dirname(sys.modules['__main__'].__file__)
urdf = rootdir + "/jaco_description/urdf/j2n6s300_twofingers.urdf"


p.connect(p.DIRECT)


a = p.loadURDF(urdf)

joint = list(map(lambda i: p.getJointInfo(a, i), range(8)))
jointnum = p.getNumJoints(a)
print(joint)