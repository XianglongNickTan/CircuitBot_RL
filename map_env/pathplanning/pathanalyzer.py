import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

from map import Map
from pathplanner import PathPlanner


class PathAnalyzer:
    def __init__(self):
        self.map = Map()
        self.path_planners = [PathPlanner(self), PathPlanner(self)]
        self.result = [PathResult(), PathResult()]
        self.min_slope = 0.5
        self.max_slope = 2

    def set_map(self, array):
        self.map.read_fromNdArray(array)

    def set_slope_value(self, min_val, max_val):
        self.min_slope = min_val
        self.max_slope = max_val

    def set_pathplan(self, pair_no, departure, destination):
        self.path_planners[pair_no].set_pathplan(self.map.get_coordination(departure[0], departure[1]), self.map.get_coordination(destination[0], destination[1]))

    def search(self):
        success1 = self.path_planners[0].search()

        if success1:
            self.path_planners[1].set_obstacles(self.path_planners[0].path)            
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
        
        print(self.result[0])
        print(self.result[1])
    
    def get_result(self, no):
        return self.result[no].success, self.result[no].path, self.result[no].cost

         # 绘制地图以及路径
    def draw_map(self, path, pole_pair):
        print("Drawing map...")
        x = np.arange(0, self.map.width, 1)
        y = np.arange(0, self.map.height, 1)
        X, Y = np.meshgrid(x, y)
        Z = self.map.dem_map
        # 填充等高线图
        plt.contourf(X, Y, Z)
        plt.contour(X, Y, Z)
        if path is not None:
        	#首先将路径节点列表转置一下,将x坐标和y坐标分别放到一行中
            path = np.transpose(path)
            print('Path Array:', path)
            plt.scatter(path[0], path[1], c='y', linewidths=2)
        if pole_pair[0] is not None:
        	# 绘制起点坐标
            plt.scatter(pole_pair[0][0], pole_pair[0][1], c='r', linewidths=6)
        if pole_pair[1] is not None:
       		 # 绘制终点坐标
            plt.scatter(pole_pair[1][0], pole_pair[1][1], c='black', linewidths=6)
        plt.show()

        # 绘制地图以及路径
    
    def draw_map_3D(self):
        x = np.arange(0, self.map.width, 1)
        y = np.arange(0, self.map.height, 1)
        X, Y = np.meshgrid(x, y)
        Z = self.map.dem_map
        
        # fig,ax = plt.subplots(subplot_kw=dict(projection='3d'),figsize=(12,10))
        # # 注意此处要手动设置一下z轴的高度,否则地图的比例会很奇怪
        # ls = LightSource(270,20)
        # rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        # # 绘制地形图
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
        #                             linewidth=0, antialiased=False, shade=False)

        fig,ax = plt.subplots(subplot_kw=dict(projection='3d'),figsize=(12,10))
        # 注意此处要手动设置一下z轴的高度,否则地图的比例会很奇怪
        ax.plot_wireframe(X, Y, Z, cmap=plt.cm.gist_earth)
        
        ax.set_zlim(0,40)

         # 绘制三维路线图
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

        # temp = np.where(np.absolute(self.map.dem_map - 0) > self.min_slope)
        # slope_pts = np.transpose(np.where(np.absolute(temp) < self.max_slope))

        # slope_ptsT = np.transpose(slope_pts)
        # Xp3 = slope_ptsT[1]
        # Yp3 = slope_ptsT[0]
        # Zp3 = [self.map.dem_map[pos[0], pos[1]] for pos in slope_pts]

        # ax.scatter(Xp3,Yp3,Zp3,c='r',s=800)
        plt.show()


class PathResult():
    def __init__(self, success = False, path = [], cost = -1):
        self.success = success
        self.path = path
        self.cost = cost

    def __str__(self):
        return str(self.success) + str(self.path) + str(self.cost)