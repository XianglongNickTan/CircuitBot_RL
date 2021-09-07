# -*- coding: utf-8 -*-

import rospy, sys
import moveit_commander
from moveit_commander import PlanningSceneInterface
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint
import math
import numpy as np
import copy
import serial
import random
import os


import actionlib
import kinova_msgs.msg


# arduino_motor = serial.Serial('/dev/ttyACM0', 38400)
# arduino_voltage = serial.Serial('/dev/ttyACM1', 9600, timeout=0.5)




""" Global variable """
arm_joint_number = 0
finger_number = 0
prefix = 'j2n6s300' + '_'
finger_maxDist = 18.9/2/1000  # max distance for one finger
finger_maxTurn = 6800  # max thread rotation for one finger
currentFingerPosition = [0.0, 0.0, 0.0]

def gripper_client(finger_positions):
	"""Send a gripper goal to the action server."""
	action_address = '/' + prefix + 'driver/fingers_action/finger_positions'

	client = actionlib.SimpleActionClient(action_address,
										  kinova_msgs.msg.SetFingersPositionAction)
	client.wait_for_server()

	goal = kinova_msgs.msg.SetFingersPositionGoal()
	goal.fingers.finger1 = float(finger_positions[0])
	goal.fingers.finger2 = float(finger_positions[1])
	# The MICO arm has only two fingers, but the same action definition is used
	if len(finger_positions) < 3:
		goal.fingers.finger3 = 0.0
	else:
		goal.fingers.finger3 = float(finger_positions[2])
	client.send_goal(goal)
	if client.wait_for_result(rospy.Duration(5.0)):
		return client.get_result()
	else:
		client.cancel_all_goals()
		rospy.logwarn('        the gripper action timed-out')
		return None


def turn_finger(finger_value_):
	""" Argument unit """

	# get absolute value
	finger_turn_command = [x/100.0 * finger_maxTurn for x in finger_value_]

	result = gripper_client(finger_turn_command)











class MoveItIkDemo:
	def __init__(self):
		# 初始化move_group的API
		joint_state_topic = ['joint_states:=/j2n6s300_driver/out/joint_state']
		moveit_commander.roscpp_initialize(joint_state_topic)
		moveit_commander.roscpp_initialize(sys.argv)

		# 初始化ROS节点
		rospy.init_node('real_jaco_moveit_fixed_grasp', anonymous=True)

		# 初始化需要使用move group控制的机械臂中的self.arm group
		self.arm = moveit_commander.MoveGroupCommander('arm')

		# 初始化需要使用move group控制的机械臂中的gripper group
		self.gripper = moveit_commander.MoveGroupCommander('gripper')

		# 获取终端link的名称
		self.end_effector_link = self.arm.get_end_effector_link()

		# 设置目标位置所使用的参考坐标系
		self.reference_frame = 'world'
		self.arm.set_pose_reference_frame(self.reference_frame)


		# 当运动规划失败后，不允许重新规划
		self.arm.allow_replanning(False)

		# 设置位置(单位：米)和姿态（单位：弧度）的允许误差
		self.arm.set_goal_position_tolerance(0.001)
		self.arm.set_goal_orientation_tolerance(0.001)
		# self.gripper.set_goal_joint_tolerance(0.001)

		# 设置允许的最大速度和加速度
		self.arm.set_max_acceleration_scaling_factor(0.2)
		self.arm.set_max_velocity_scaling_factor(0.5)

		# 初始化场景对象
		scene = PlanningSceneInterface()
		rospy.sleep(1)



		# # 控制机械臂先回到初始化位置，手爪打开
		# self.arm.set_named_target('Home')
		# self.arm.go()
		# self.gripper.set_named_target('Open')
		# self.gripper.go()
		rospy.sleep(1)

		self.line_cont = 20
		self.circle_cont = 40
		self.square_cont = 10
		self.triangle_cont = 10

		self.x_offset = -2
		self.y_offset = -53

		self.target_pose = PoseStamped()
		self.target_pose.header.frame_id = self.reference_frame
		self.target_pose.pose.position.z = 0.03

		# self.target_pose.pose.orientation.x = 0.14578
		# self.target_pose.pose.orientation.y = 0.98924
		# self.target_pose.pose.orientation.z = -0.0085346
		# self.target_pose.pose.orientation.w = 0.0084136

		self.target_pose.pose.orientation.x = 0.0896246507764
		self.target_pose.pose.orientation.y = 0.994087159634
		self.target_pose.pose.orientation.z = -0.0611566230655
		self.target_pose.pose.orientation.w = -0.00424344977364

		### gripper control ###
		self.prefix = 'j2n6s300' + '_'
		self.finger_maxTurn = 6800  # max thread rotation for one finger



	def talker(self, center_x):
		pub = rospy.Publisher('center_x', Float32, queue_size=10)
		# rospy.init_node('talker', anonymous=True)
		rate = rospy.Rate(10)  # 10hz
		for _ in range(10):
			rospy.loginfo(center_x)
			pub.publish(center_x)
			rate.sleep()


	def gripper_client(self, finger_positions):
		"""Send a gripper goal to the action server."""
		action_address = '/' + self.prefix + 'driver/fingers_action/finger_positions'

		client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.SetFingersPositionAction)
		client.wait_for_server()

		goal = kinova_msgs.msg.SetFingersPositionGoal()
		goal.fingers.finger1 = float(finger_positions[0])
		goal.fingers.finger2 = float(finger_positions[1])
		# The MICO arm has only two fingers, but the same action definition is used
		if len(finger_positions) < 3:
			goal.fingers.finger3 = 0.0
		else:
			goal.fingers.finger3 = float(finger_positions[2])
		client.send_goal(goal)
		if client.wait_for_result(rospy.Duration(5.0)):
			return client.get_result()
		else:
			client.cancel_all_goals()
			rospy.logwarn('        the gripper action timed-out')
			return None

	def turn_finger(self, finger_value_):
		""" Argument unit """

		# get absolute value
		finger_turn_command = [x / 100.0 * self.finger_maxTurn for x in finger_value_]
		result = self.gripper_client(finger_turn_command)



	def home_robot(self):
		self.arm.set_named_target('Home')
		self.arm.go()

	def open_finger(self):
		pass

	def move_to(self, point):

		# point[0] = (point[0] + self.x_offset) * 0.01
		# point[1] = (point[1] + self.x_offset) * 0.01

		self.target_pose.header.stamp = rospy.Time.now()
		self.target_pose.pose.position.x = point[0]
		self.target_pose.pose.position.y = point[1]
		self.target_pose.pose.position.z = point[2]


		self.arm.set_start_state_to_current_state()
		self.arm.set_pose_target(self.target_pose, self.end_effector_link)
		traj = self.arm.plan()
		self.arm.execute(traj)



	def draw_line(self, xy_start_pos, xy_end_pos):
		waypoints = []

		xy_start_pos[0] = (xy_start_pos[0] + self.x_offset) * 0.01
		xy_start_pos[1] = (xy_start_pos[1] + self.y_offset) * 0.01
		xy_end_pos[0] = (xy_end_pos[0] + self.x_offset) * 0.01
		xy_end_pos[1] = (xy_end_pos[1] + self.y_offset) * 0.01

		stay_point = [0, -5]
		stay_point[0] = (stay_point[0] + self.x_offset) * 0.01
		stay_point[1] = (stay_point[1] + self.y_offset) * 0.01

		self.target_pose.header.stamp = rospy.Time.now()
		self.target_pose.pose.position.x = stay_point[0]
		self.target_pose.pose.position.y = stay_point[1]

		self.arm.set_start_state_to_current_state()
		self.arm.set_pose_target(self.target_pose, self.end_effector_link)
		traj = self.arm.plan()
		self.arm.execute(traj)

		self.target_pose.header.stamp = rospy.Time.now()
		self.target_pose.pose.position.x = xy_start_pos[0]
		self.target_pose.pose.position.y = xy_start_pos[1]

		self.arm.set_start_state_to_current_state()
		self.arm.set_pose_target(self.target_pose, self.end_effector_link)
		traj = self.arm.plan()
		self.arm.execute(traj)
		rospy.sleep(2)


		wpose = self.arm.get_current_pose()
		# wpose.pose.orientation.x = 0.14578
		# wpose.pose.orientation.y = 0.98924
		# wpose.pose.orientation.z = -0.00853
		# wpose.pose.orientation.w = 0.00841
		# wpose.pose.position.z = 0.03
		waypoints.append(copy.deepcopy(wpose.pose))

		# self.init_upright_path_constraints(wpose)
		# self.enable_upright_path_constraints()


		# for _ in range(2):
		for t in range(self.line_cont):
			wpose.pose.position.x = ((self.line_cont - t) * xy_start_pos[0] + t * xy_end_pos[0]) / self.line_cont
			wpose.pose.position.y = ((self.line_cont - t) * xy_start_pos[1] + t * xy_end_pos[1]) / self.line_cont
			waypoints.append(copy.deepcopy(wpose.pose))

		(plan, fraction) = self.arm.compute_cartesian_path(
			waypoints,
			0.01,             # SUPER IMPORTANT PARAMETER FOR VELOCITY CONTROL !!!!!
			0.0
		)


		# arduino_motor.write('2')
		# self.arm.execute(traj)
		# arduino_motor.write('0')
		# self.arm.stop()

		arduino_motor.write('2')
		self.arm.execute(plan)
		arduino_motor.write('0')
		self.arm.stop()

		# self.disable_upright_path_constraints()

		rospy.sleep(2)


def test(xy_center_pos):
	######################## For testing ########################

	voltage = xy_center_pos[0] - xy_center_pos[1]**2 - xy_center_pos[2]**2 + xy_center_pos[3] + xy_center_pos[4]**2 - xy_center_pos[5]
	voltage_file = open("../control/voltage.txt", "w")
	# print ("-------------------------")
	# print ("file name: ", voltage_file.name)
	# print ("-------------------------")
	voltage_file.write(str(voltage))
	voltage_file.close()
	print("voltage:", voltage)

######################## For testing ########################


if __name__ == "__main__":
	demo = MoveItIkDemo()

	point = [0, -0.7, 0.2]
	angle = [0, 0, 0]
	# demo.home_robot()
	demo.turn_finger(angle)
	demo.move_to(point)


	# point = [0, -0.7, 0.1]
	# angle = [80, 80, 0]
	# # demo.home_robot()
	# demo.move_to(point)
	# demo.turn_finger(angle)
	#
	#
	# point = [0, -0.5, 0.5]
	# angle = [0, 0, 0]
	# # demo.home_robot()
	# demo.move_to(point)
	# demo.turn_finger(angle)



	moveit_commander.roscpp_shutdown()
	moveit_commander.os._exit(0)
