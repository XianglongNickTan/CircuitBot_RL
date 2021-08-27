import numpy as np
import math
from pathnodes.pathnode import PathNode

# F(n)=g(n)+h(n)
class PathPlanner:
    def __init__(self):
        self.slope_val = 0
        self.map = None
        self.source = None
        self.destination = None
        self.target = None
        self.node_searched = 0
        self.obstacles = []
        self.path = []
        self.root = None

    def update(self, slope_val, map, source, target, obstacles = []):
        self.slope_val = slope_val
        self.map = map
        self.source = source
        self.destination = None
        self.target = target
        self.node_searched = 0
        self.obstacles = obstacles if len(obstacles) != 0 else []
        self.root = PathNode(self.slope_val, self.map, self.source, self.destination, self.target, obstacles = self.obstacles)

    def search(self):
        self.root.start_search()
        self.path = self.root.get_path()
        return self.path


