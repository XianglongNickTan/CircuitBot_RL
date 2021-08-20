import math
import os
import sys
import numpy as np

rootdir = os.path.dirname(sys.modules['__main__'].__file__)
jacoUrdf = rootdir + "/jaco_description/urdf/j2n6s300.urdf"
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
            spawn_pos,
            self.p.getQuaternionFromEuler([0, 0, -math.pi / 2]),
            flags=self.p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)


        ## workspace
        self.reach_low = np.array(reach_low) + spawn_pos
        self.reach_low[2] = reach_low[2]
        self.reach_high = np.array(reach_high) + spawn_pos

        ## joint info
        self.numJoints = self.p.getNumJoints(self.armId)
        assert self.numJoints
        self.jointVelocities = [0] * self.numJoints

        ## grip variable ##
        self.gripperOpen = 1
        self.gripperClose = 1.2

        self.FingerAngle = [
            self.p.getJointState(self.armId, i)[0] for i in [7, 9]
        ]

        ## goal variables ##

        self.goalOrientation = [0, -math.pi, -math.pi / 2]
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

        self.p.resetJointState(self.armId, 7, 1)    # finger 1 base joint
        self.p.resetJointState(self.armId, 9, 1)    # finger 2 base joint

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
                250, 193.235290527, 52.0588226318, 348.0, -314.522735596,
                238.5, 0.0
            ]))

        self.IKInfo["solver"] = 0
        self.IKInfo["jointDamping"] = [0.005] * self.numJoints
        self.IKInfo["endEffectorLinkIndex"] = 6

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
        joint_poses = self.compute_ik_poses()

        # Set all body joints.
        for i in range(len(joint_poses)):
            self.p.setJointMotorControl2(
                bodyIndex=self.armId,
                jointIndex=i,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.03,
                velocityGain=1)

        # Set gripper joints.
        # self.p.setJointMotorControl2(
        #     bodyIndex=self.armId,
        #     jointIndex=7,
        #     controlMode=self.p.POSITION_CONTROL,
        #     targetPosition= self.goalFinger,
        #     targetVelocity=0,
        #     force=500,
        #     positionGain=0.03,
        #     velocityGain=1)
        # self.p.setJointMotorControl2(
        #     bodyIndex=self.armId,
        #     jointIndex=9,
        #     controlMode=self.p.POSITION_CONTROL,
        #     targetPosition= self.goalFinger,
        #     targetVelocity=0,
        #     force=500,
        #     positionGain=0.03,
        #     velocityGain=1)

        # Save whether the arm has reached IK goal.


        self.goalReached = all(map(
                lambda joint: abs(self.p.getJointState(self.armId, joint)[
                              0] - joint_poses[joint]) < self.goalEpsilon, range(6)
            )
        )

    def step_simulation(self):
        """Step the simulation."""
        self.ik_step()
        # self.check_contacts()


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

        gripper_reached = [
            abs(self.p.getJointState(self.armId, i)[0] - finger_angle) <
            self.goalEpsilon for i in [7, 9]
        ]

        self.goalReached = self.goalReached and all(gripper_reached)


    ############ high-level action control ##########

    def print_circuit(self):
        """ Circuit path print, move arm horizontally, display trajectory"""
        pass

    def pick_place_object(self, init_pos, end_pos):
        """ pick and place object"""
