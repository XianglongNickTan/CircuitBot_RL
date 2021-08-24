import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm

class Map:
    def __init__(self):
        self.width = 1
        self.height = 1
        self.map = np.zeros((self.height, self.width), dtype = int)
        self.dem_map = None
        self.grid_size = 1

    # 从文件读取地图
    def read_fromfile(self, file_path, grid_size=1.0):
        print('Loading DEM map...')
        fin = open(file_path, 'r')
        new_map = []
        for row in fin.readlines():
            heights = [float(point) for point in row.strip().split(' ')]
            new_map.append(heights)
        # new_map.reverse()
        # 将地理栅格模型的y轴反转
        new_map = np.array(new_map)
        print('mean:', new_map.mean())
        self.dem_map = new_map
        print(new_map.shape)

        self.height, self.width = new_map.shape
        print(self.height)
        print(self.width)
        self.grid_size = grid_size
        self.map = np.zeros((self.height, self.width), dtype=int)
        print(f'DEM map loaded. Width={self.width},height={self.height},grid size={self.grid_size}.')
    
    # 从文件读取地图
    def read_fromNdArray(self, map, grid_size=1.0):
        print('Loading DEM map...')
        print('mean:', map.mean())
        
        self.dem_map = np.flipud(map)
        self.height, self.width = map.shape
        self.grid_size = grid_size
        self.map = np.zeros((self.height, self.width), dtype=int)
        print(f'DEM map loaded. Width={self.width},height={self.height},grid size={self.grid_size}.')


