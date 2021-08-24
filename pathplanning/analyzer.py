from map import Map
from pathplanner import PathPlanner
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

class Analyzer:
    def __init__(self):
        self.map = Map()
        self.planner = PathPlanner()
        self.source = [0,1]
        self.dest = [0,2]
        self.path = []
    
    def updateMap(self, array):
        self.map.read_fromNdArray(array)

    def update(self, source, dest):
        self.source = source
        self.dest = dest

    def search(self):
        self.planner.update(self.map, self.source, self.dest)
        self.planner.start_search()
        self.path = self.planner.path

    def render(self): 
        self.map

     # 绘制地图以及路径
    def draw_map(self):
        print("Drawing map...")
        x = np.arange(0, self.map.width, 1)
        y = np.arange(0, self.map.height, 1)
        X, Y = np.meshgrid(x, y)
        Z = self.map.dem_map
        # 填充等高线图
        plt.contourf(X, Y, Z)
        plt.contour(X, Y, Z)
        if self.path is not None:
        	#首先将路径节点列表转置一下,将x坐标和y坐标分别放到一行中
            path = np.transpose(self.path)
            print('Path Array:', path)
            plt.scatter(path[0], path[1], c='y', linewidths=2)
        if self.source is not None:
        	# 绘制起点坐标
            plt.scatter(self.source[0], self.source[1], c='r', linewidths=6)
        if self.dest is not None:
       		 # 绘制终点坐标
            plt.scatter(self.dest[0], self.dest[1], c='black', linewidths=6)
        plt.show()

        # 绘制地图以及路径
    def draw_map3D(self):
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
        pathT = np.transpose(self.path)
        Xp = pathT[0]
        Yp = pathT[1]
        Zp = [self.map.dem_map[pos[1], pos[0]] for pos in self.path]
        ax.scatter(Xp,Yp,Zp,c='r',s=200)
        plt.show()