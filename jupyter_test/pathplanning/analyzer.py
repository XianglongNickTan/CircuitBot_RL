import numpy as np
from map import Map
from pathplanner import PathPlanner
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

class Analyzer:
    def __init__(self):
        self.map = Map()
        self.planners = [PathPlanner(), PathPlanner()]
        self.pole_pairs = [[],[]]
        self.paths = [[],[]]
        self.slope_val = 0.5
    
    def updateMap(self, array):
        self.map.read_fromNdArray(array)

    def update_slope_value(self, val):
        self.slope_val = val

    def update_pole_pair(self, pair_no = 0, source = [0,0], target = [0,0]):
        self.pole_pairs[pair_no] = [self.map.get_coordination(source[0], source[1]), self.map.get_coordination(target[0], target[1])]

    def search(self):
        self.planners[0].update(self.slope_val, self.map, self.pole_pairs[0][0], self.pole_pairs[0][1])
        self.paths[0] = self.planners[0].search()

        print("pathhhhhhhhhhhhhhhhhhhhhhhhhh2")
        print("pathhhhhhhhhhhhhhhhhhhhhhhhhh2")
        print("pathhhhhhhhhhhhhhhhhhhhhhhhhh2")
        print("pathhhhhhhhhhhhhhhhhhhhhhhhhh2")
        print("pathhhhhhhhhhhhhhhhhhhhhhhhhh2")
        print("pathhhhhhhhhhhhhhhhhhhhhhhhhh2")

        self.planners[1].update(self.slope_val, self.map, self.pole_pairs[1][0], self.pole_pairs[1][1], self.paths[0])
        self.paths[1] = self.planners[1].search()

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
        
        ax.set_zlim(0,4)

         # 绘制三维路线图
        pathT = np.transpose(self.paths[0])  # [[1,2],[1,2],[1,2]...]
        Xp = pathT[0] #[[1,1,1,1,1], [2,2,2,2]...]
        Yp = pathT[1]
        Zp = [self.map.dem_map[pos[1], pos[0]] for pos in self.paths[0]]
        ax.scatter(Xp,Yp,Zp,c='g',s=200)

        pathT2 = np.transpose(self.paths[1])
        Xp2 = pathT2[0]
        Yp2 = pathT2[1]
        Zp2 = [self.map.dem_map[pos[1], pos[0]] for pos in self.paths[1]]
        ax.scatter(Xp2,Yp2,Zp2,c='b',s=200)

        slope_pts = np.transpose(np.where(np.absolute(self.map.dem_map - 0) == self.slope_val))
        slope_ptsT = np.transpose(slope_pts)
        Xp3 = slope_ptsT[1]
        Yp3 = slope_ptsT[0]
        Zp3 = [self.map.dem_map[pos[0], pos[1]] for pos in slope_pts]

        ax.scatter(Xp3,Yp3,Zp3,c='r',s=800)
        plt.show()