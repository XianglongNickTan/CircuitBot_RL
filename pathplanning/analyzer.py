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
    
    def updateMap(self, array):
        self.map.read_fromNdArray(array)

    def update_pole_pair(self, pair_no = 0, source = [0,0], dest = [0,0]):
        self.pole_pairs[pair_no] = [source, dest]

    def search(self):
        self.planners[0].update(self.map, self.pole_pairs[0][0], self.pole_pairs[0][1])
        self.planners[0].start_search()
        self.paths[0] = self.planners[0].path

        self.planners[1].update(self.map, self.pole_pairs[1][0], self.pole_pairs[1][1], self.planners[0].path)
        self.planners[1].start_search()
        self.paths[1] = self.planners[1].path

    def render(self): 
        self.map

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

        combined_path = self.paths[0] + self.paths[1]

         # 绘制三维路线图
        pathT = np.transpose(combined_path)
        Xp = pathT[0]
        Yp = pathT[1]
        Zp = [self.map.dem_map[pos[1], pos[0]] for pos in combined_path]
        ax.scatter(Xp,Yp,Zp,c='r',s=200)
        plt.show()