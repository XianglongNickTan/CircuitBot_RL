import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

from map_env.pathplanning.map import Map
from map_env.pathplanning.pathplanner import PathPlanner

class PathAnalyzer:
    def __init__(self):
        self.map = Map()
        self.path_planners = [PathPlanner(self), PathPlanner(self)]
        self.result = [PathResult(), PathResult()]
        self.min_slope = 0.2
        self.max_slope = 2

    def set_map(self, array):
        self.map.read_fromNdArray(array)

    def set_slope_value(self, min_val, max_val):
        self.min_slope = min_val
        self.max_slope = max_val

    def set_pathplan(self, pair_no, departure, destination):
        self.path_planners[pair_no].set_pathplan(self.map.get_coordination(departure[0], departure[1]), self.map.get_coordination(destination[0], destination[1]))

    def add_obstacles(self, new_obstacles):
        self.path_planners[0].add_obstacles(new_obstacles)

    def set_obstacles(self, new_obstacles):
        self.path_planners[0].set_obstacles(new_obstacles)

    def clear_obstacles(self, obstacles):
        self.path_planners[0].clear_obstacles()

    def search(self):
        success1 = self.path_planners[0].search()

        if success1:
            self.path_planners[1].add_obstacles(self.path_planners[0].obstacles + self.path_planners[0].path)            
            success2 = self.path_planners[1].search()
            if success2:
                self.result[0] = PathResult(True, self.path_planners[0].path, self.path_planners[0].cost)
                self.result[1] = PathResult(True, self.path_planners[1].path, self.path_planners[1].cost)
            else:
                self.result[0] = PathResult(True, self.path_planners[0].path, self.path_planners[0].cost)
                self.result[1] = PathResult(False, [], -1)

        else:
            self.result[0] = PathResult(False, [], -1)
            self.result[1] = PathResult(False, [], -1)

        self.path_planners[0].obstacles.clear()
        self.path_planners[1].obstacles.clear()
    
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
        
        fig,ax = plt.subplots(subplot_kw=dict(projection='3d'),figsize=(12,10))
        ax.plot_wireframe(X, Y, Z, cmap=plt.cm.gist_earth)
        
        ax.set_zlim(0,40)
        pathT = np.transpose(self.path_planners[0].path)  # [[1,2],[1,2],[1,2]...]
        Xp = pathT[0] #[[1,1,1,1,1], [2,2,2,2]...]
        Yp = pathT[1]
        Zp = [self.map.dem_map[pos[1], pos[0]] for pos in self.path_planners[0].path]
        ax.scatter(Xp,Yp,Zp,c='g',s=200)

        pathT2 = np.transpose(self.path_planners[1].path)
        Xp2 = pathT2[0]
        Yp2 = pathT2[1]
        Zp2 = [self.map.dem_map[pos[1], pos[0]] for pos in self.path_planners[1].path]
        ax.scatter(Xp2,Yp2,Zp2,c='b',s=200)

        plt.show()

class PathResult():
    def __init__(self, success = False, path = [], cost = -1):
        self.success = success
        self.path = path
        self.cost = cost

    def __str__(self):
        return str(self.success) + str(self.path) + str(self.cost)