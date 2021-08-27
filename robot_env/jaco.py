import math
import os
import sys
import numpy as np
import time


rootdir = os.path.dirname(sys.modules['__main__'].__file__)
jacoUrdf = rootdir + "/jaco_description/urdf/j2n6s300_twofingers.urdf"
# jacoUrdf = rootdir + "/mico_description/urdf/mico.urdf"
np.set_printoptions(precision=2, floatmode='fixed', suppress=True)


class Jaco:
    def __init__(self,
                 p,  # Simulator object
                 spawn_pos=(0, 0, 0),  # Position where to spawn the arm.
                 reach_low=(-1, -1, 0),  # Lower limit of the arm workspace (might be used for safety).
                 reach_high=(1, 1, 1),  # Higher limit of the arm workspace.
                 randomize_arm=False,  # Whether arm initial position should be randomized.
                 urdf=jacoUrdf  # Where to load the arm definition.
                 ):
        self.p = p

        ## load urdf
        self.armId = self.p.loadURDF(
            urdf,
            basePosition=spawn_pos,
            baseOrientation=self.p.getQuaternionFromEuler([0, 0, -math.pi / 2]),
            useFixedBase=1)
            # flags=self.p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)


        ## workspace
        self.reach_low = np.array(reach_low) + spawn_pos
        self.reach_low[2] = reach_low[2]
        self.reach_high = np.array(reach_high) + spawn_pos

        ## joint info
        self.numJoints = self.p.getNumJoints(self.armId)
        assert self.numJoints
        self.jointVelocities = [0] * self.numJoints

        ## grip control ##
        self.gripperOpen = 0.5
        self.gripperClose = 1.3

        self.cur_FingerAngle = [
            self.p.getJointState(self.armId, i)[0] for i in [7, 9]
        ]

        self.gripperConstraints = []

        ## goal variables ##
        self.goalOrientation = [0, -math.pi, math.pi / 3]

        # self.goalOrientation = [0, -math.pi, -math.pi / 2]

        self.goalReached = False
        self.goalEpsilon = 0.1

        ## initialisation ##
        self.get_ik_information()

        if randomize_arm:
            init_pos = np.random.normal(self.IKInfo["restPoses"], 0.05)
        else:
            init_pos = self.IKInfo["restPoses"]

        for i in range(len(self.IKInfo["restPoses"])):
            self.p.resetJointState(self.armId, i, init_pos[i])

        self.p.resetJointState(self.armId, 7, self.gripperOpen)    # finger 1 base joint
        self.p.resetJointState(self.armId, 9, self.gripperOpen)    # finger 2 base joint

        grip_pos = np.array(
            self.p.getLinkState(self.armId, 6, computeLinkVelocity=1)[0])

        self.goalPosition = grip_pos


    ######### ik control ##########

    def get_ik_information(self):
        """ Finds the values for the IK solver. """
        joint_information = list(
            map(lambda i: self.p.getJointInfo(self.armId, i),
                range(self.numJoints)))
        self.IKInfo = {}
        assert all(
            [len(joint_information[i]) == 17 for i in range(self.numJoints)])
        self.IKInfo["lowerLimits"] = list(
            map(lambda i: joint_information[i][8], range(7)))
        self.IKInfo["upperLimits"] = list(
            map(lambda i: joint_information[i][9], range(7)))

        self.IKInfo["jointRanges"] = list(
            map(lambda i: joint_information[i][9] - joint_information[i][8],
                range(7)))

        self.IKInfo["restPoses"] = list(
            map(math.radians, [
                0, 0, 0, 0, 0,
                0
            ]))

        # self.IKInfo["restPoses"] = list(
        #     map(math.radians, [
        #         250, 193.235290527, 52.0588226318, 348.0, -314.522735596,
        #         238.5, 0.0, 0
        #     ]))

        self.IKInfo["solver"] = 0
        self.IKInfo["jointDamping"] = [0.005] * self.numJoints
        self.IKInfo["endEffectorLinkIndex"] = 6

    def get_endEffector_pos(self):
        grip_pos = np.array(
            self.p.getLinkState(self.armId, 6, computeLinkVelocity=1)[0])
        return grip_pos

    def compute_ik_poses(self):
        """ Use the IK solver to compute goal joint poses to achieve IK goal."""
        joint_poses = self.p.calculateInverseKinematics(
            self.armId,
            targetPosition=self.goalPosition,
            targetOrientation=self.p.getQuaternionFromEuler(
                self.goalOrientation),
            **self.IKInfo)
        return joint_poses

    def ik_step(self):
        """ Set arm to calculated pose."""
        self.joint_poses = self.compute_ik_poses()

        # Set all body joints.
        for i in range(len(self.joint_poses)):
            self.p.setJointMotorControl2(
                bodyIndex=self.armId,
                jointIndex=i,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=self.joint_poses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.03,
                velocityGain=1)

        self.goalReached = all(map(
                lambda joint: abs(self.p.getJointState(self.armId, joint)[
                              0] - self.joint_poses[joint]) < self.goalEpsilon, range(6)
            )
        )

    def step_simulation(self):
        """Step the simulation."""
        while not self.goalReached:
            self.check_goal_reach()
            self.p.stepSimulation()
            time.sleep(1. / 240.)     # set time interval for visulaization


    #### Save whether the arm has reached IK goal. ###
    def check_goal_reach(self):

        self.goalReached = all(map(
                lambda joint: abs(self.p.getJointState(self.armId, joint)[
                              0] - self.joint_poses[joint]) < self.goalEpsilon, range(6)
            )
        )

        # gripper_reached = [
        #     abs(self.p.getJointState(self.armId, i)[0] - self.finger_angle) <
        #     self.goalEpsilon for i in [7, 9]
        # ]
        #
        # self.goalReached = self.goalReached and all(gripper_reached)


    ############ grasp control ##########

    def gripper_control(self, state):
        """Control the open or close of the gripper
            state == true : open   otherwise: close"""

        if state:
            finger_angle = self.gripperOpen
        else:
            finger_angle = self.gripperClose

        self.p.setJointMotorControl2(
            bodyIndex=self.armId,
            jointIndex=7,
            controlMode=self.p.POSITION_CONTROL,
            targetPosition= finger_angle,
            targetVelocity=0,
            force=500,
            positionGain=0.03,
            velocityGain=1)
        self.p.setJointMotorControl2(
            bodyIndex=self.armId,
            jointIndex=9,
            controlMode=self.p.POSITION_CONTROL,
            targetPosition= finger_angle,
            targetVelocity=0,
            force=500,
            positionGain=0.03,
            velocityGain=1)

        # gripper_reached = [
        #     abs(self.p.getJointState(self.armId, i)[0] - self.finger_angle) <
        #     self.goalEpsilon for i in [7, 9]
        # ]
        #
        # self.goalReached = self.goalReached and all(gripper_reached)


    def create_gripper_constraints(self, object_id):
        # if not self.gripperConstraints:
        #     self.gripperConstraints.append(
        #         self.p.createConstraint(
        #             parentBodyUniqueId=self.armId,
        #             parentLinkIndex=6,
        #             childBodyUniqueId=object_id,
        #             childLinkIndex=-1,
        #             parentFramePosition=[0, 0, 0],
        #             childFramePosition=[0, 0, 0],
        #             jointAxis=[1, 1, 1],
        #             jointType=self.p.JOINT_POINT2POINT,
        #         ))

        self.gripperConstraints.append(
            self.p.createConstraint(
                parentBodyUniqueId=self.armId,
                parentLinkIndex=6,
                childBodyUniqueId=object_id,
                childLinkIndex=-1,
                parentFramePosition=[-0.03, 0, 0.03],
                childFramePosition=[0, 0, 0],
                jointAxis=[1, 1, 1],
                jointType=self.p.JOINT_FIXED,
                parentFrameOrientation=self.p.getQuaternionFromEuler([0, 0, -math.pi / 2.35])
            ))


    def remove_gripper_constraints(self):
        if not self.gripperConstraints:
            return
        for constraint in self.gripperConstraints:
            self.p.removeConstraint(constraint)
        self.gripperConstraints = []

    ############ ik move ############
    def move_to(self, pos, ori):
        self.goalPosition = pos
        self.goalOrientation = ori
        self.ik_step()
        self.step_simulation()

    def pick(self):
        self.gripper_control(False)
        # self.step_simulation()
        for _ in range(20):
            self.p.stepSimulation()

    def place(self):
        self.gripper_control(True)
        # self.step_simulation()
        for _ in range(20):
            self.p.stepSimulation()



    ############ high-level action control ##########

    def print_circuit(self):
        """ Circuit path print, move arm horizontally, display trajectory"""
        pass

    def pick_place_object(self, init_pos, init_ori, end_pos, end_ori):
        """ pick and place object"""

        self.move_to(init_pos, init_ori)
        self.pick()
        self.move_to(end_pos, end_ori)
        self.place()




