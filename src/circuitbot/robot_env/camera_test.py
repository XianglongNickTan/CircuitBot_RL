import pybullet as p
import time
import cv2
import pybullet_data
import numpy as np
import os, sys


rootdir = os.path.dirname(sys.modules['__main__'].__file__)
urdf = rootdir + "/robot_env/jaco_description/urdf/j2n6s300_twofingers.urdf"


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF(urdf,startPos, startOrientation,useFixedBase=1)


trayId = p.loadURDF("table_square/table_square.urdf")

#set the center of mass frame (loadURDF sets base link frame)
# startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos,
# startOrientation)

# viewMatrix = p.computeViewMatrix([-1.05, 0, 0.68], [0.1, 0, 0],
#                                       [-0.5, 0, 1])
# projMatrix = p.computeProjectionMatrixFOV(
#     fov=45, aspect=4. / 3., nearVal=0.01, farVal=2.5)

viewMatrix = p.computeViewMatrix([0, 0.3, 2], [0, 0.3, -1],
                                      [0, 1, 0])
projMatrix = p.computeProjectionMatrixFOV(
    fov=45, aspect=4. / 3., nearVal=0.01, farVal=2.5)


light = {
    "diffuse": 0.4,
    "ambient": 0.5,
    "spec": 0.2,
    "dir": [10, 10, 100],
    "col": [1, 1, 1]}

def render(mode='human'):
    width, height = 106, 84
    img = p.getCameraImage(
        width,
        height,
        viewMatrix,
        projMatrix,
        shadow=1,
        lightAmbientCoeff=light["ambient"],
        lightDiffuseCoeff=light["diffuse"],
        lightSpecularCoeff=light["spec"],
        lightDirection=light["dir"],
        lightColor=light["col"]
    )

    # img = p.getCameraImage(
    #     width,
    #     height
    # )


    rgb = np.array(img[2], dtype=np.float).reshape(height, width, 4) / 255
    rgb[:, :, 3], rgb[:, :, 2] = rgb[:, :, 2], rgb[:, :, 0]
    rgb[:, :, 0] = rgb[:, :, 3]
    rgb = rgb[:, 11:-11, :]
    if mode == 'rgb_array':
        # rgb[:,:,3] = d # No depth
        return rgb[:, :, 0:3]
    elif mode == 'human':
        cv2.imshow("test", rgb[:, :, 0:3])
        cv2.waitKey(1)


width, height = 106, 84


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

cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)





p.disconnect()