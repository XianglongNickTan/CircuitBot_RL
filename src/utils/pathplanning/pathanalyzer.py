import numpy as np
import time

from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

from utils.pathplanning.map import Map
from utils.pathplanning.pathplanner import PathPlanner
from utils.pathplanning.pathplanner import PathResult


class PathAnalyzer:
	def __init__(self):
		self.map = Map()
		self.path_planners = [PathPlanner(self), PathPlanner(self)]
		self.result = [PathResult(), PathResult()]
		self.min_slope = 0.3
		self.max_slope = 0.7

	def set_map(self, array):
		self.map.read_fromNdArray(array)

	def set_slope_value(self, min_val, max_val):
		self.min_slope = min_val
		self.max_slope = max_val

	def set_pathplan(self, pair_no, departure, destination):
		self.path_planners[pair_no].set_pathplan(self.map.get_coordination(departure[0], departure[1]),
		                                         self.map.get_coordination(destination[0], destination[1]))

	def add_obstacle(self, new_obstacle):
		self.path_planners[0].add_obstacle(self.map.get_coordination(new_obstacle[0], new_obstacle[1]))

	def add_obstacles(self, new_obstacles):
		for new_obstacle in new_obstacles:
			pts = self.map.get_coordination(new_obstacle[0], new_obstacle[1])
			pts.z = 0
			self.path_planners[0].add_obstacle(pts)
	
	def add_obstacle_coordinations(self, new_obstacles):
		self.path_planners[0].add_obstacles(new_obstacles)

	def clear_obstacles(self):
		self.path_planners[0].clear_obstacles()
		self.path_planners[1].clear_obstacles()

	def search(self):
		last = time.time()

		result_1 = self.path_planners[0].search()
		result_2 = PathResult()

		if result_1.success:
			self.path_planners[1].add_obstacles(self.path_planners[0].obstacles)
			for new_obstacle in result_1.path:
				self.path_planners[1].add_obstacle(self.map.get_coordination(new_obstacle[0], new_obstacle[1]))
			result_2 = self.path_planners[1].search()

		self.result[0] = result_1
		self.result[1] = result_2

		print(time.time() - last)
		# self.draw_map_3D()
		self.reset()

	def search_1(self):

		result_1 = self.path_planners[0].search()
		result_2 = PathResult()

		if result_1.success:
			print(self.path_planners[0].obstacles)
			self.path_planners[1].add_obstacles(self.path_planners[0].obstacles)
			for new_obstacle in result_1.path:
				self.path_planners[1].add_obstacle(self.map.get_coordination(new_obstacle[0], new_obstacle[1]))

		self.result[0] = result_1
		self.result[1] = result_2

	def search_2(self):
		result_2 = PathResult()

		if self.result[0].success:
			result_2 = self.path_planners[1].search()

		self.result[1] = result_2
		self.draw_map_3D()


	def reset(self):
		self.clear_obstacles()
            
	def get_result(self, no):
		return self.result[no].success, self.result[no].path, self.result[no].cost

	def draw_map(self, path, pole_pair):
		x = np.arange(0, self.map.width, 1)
		y = np.arange(0, self.map.height, 1)
		X, Y = np.meshgrid(x, y)
		Z = self.map.dem_map

		plt.contourf(X, Y, Z)
		plt.contour(X, Y, Z)
		if path is not None:
			path = np.transpose(path)
			plt.scatter(path[0], path[1], c='y', linewidths=2)
		if pole_pair[0] is not None:
			plt.scatter(pole_pair[0][0], pole_pair[0][1], c='r', linewidths=6)
		if pole_pair[1] is not None:
			plt.scatter(pole_pair[1][0], pole_pair[1][1], c='black', linewidths=6)
		plt.show()

	def draw_map_3D(self):
		x = np.arange(0, self.map.width, 1)
		y = np.arange(0, self.map.height, 1)
		X, Y = np.meshgrid(x, y)
		Z = self.map.dem_map

		fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(12, 10))
		ax.plot_wireframe(X, Y, Z, cmap=plt.cm.gist_earth)
		
		ax.set_zlim(0, 40)

		if self.result[0].success:
			pathT = np.transpose(self.result[0].path)  # [[1,2],[1,2],[1,2]...]
			Xp = pathT[0]  # [[1,1,1,1,1], [2,2,2,2]...]
			Yp = pathT[1]
			Zp = [self.map.dem_map[pos[1], pos[0]] for pos in self.result[0].path]
			ax.scatter(Xp, Yp, Zp, c='g', s=200)

		if self.result[1].success:
			pathT2 = np.transpose(self.result[1].path)
			Xp2 = pathT2[0]
			Yp2 = pathT2[1]
			Zp2 = [self.map.dem_map[pos[1], pos[0]] for pos in self.result[1].path]
			ax.scatter(Xp2, Yp2, Zp2, c='b', s=200)
		
		# path3 = []
		# for i in self.path_planners[0].obstacles:
		# 	if i.y < 80 and i.x < 60:
		# 		path3.append((i.x, i.y))
		# pathT3 = np.transpose(path3)
		# Xp3 = pathT3[0]
		# Yp3 = pathT3[1]
		# Zp3 = [self.map.dem_map[pos[1], pos[0]] for pos in path3]
		# ax.scatter(Xp3, Yp3, Zp3, c='r', s=5)

		plt.show()

	def draw_map_3D_only(self):
		x = np.arange(0, self.map.width, 1)
		y = np.arange(0, self.map.height, 1)
		X, Y = np.meshgrid(x, y)
		Z = self.map.dem_map

		fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(12, 10))
		ax.plot_wireframe(X, Y, Z, cmap=plt.cm.gist_earth)

		diffs = np.absolute(self.map.dem_map - 0)

		temp = np.where((self.min_slope < diffs) & (diffs < self.max_slope))
		slope_pts = np.transpose(temp)

		slope_ptsT = np.transpose(slope_pts)
		Xp3 = slope_ptsT[1]
		Yp3 = slope_ptsT[0]
		Zp3 = [self.map.dem_map[pos[0], pos[1]] for pos in slope_pts]

		ax.scatter(Xp3, Yp3, Zp3, c='r', s=800)

		plt.show()