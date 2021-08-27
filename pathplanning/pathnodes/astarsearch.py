import math

class AStarSearch():
    def __init__(self, parent = None):
        self.parent = parent

        self.slope_val = 0
        self.map = None
        self.width = 0
        self.height = 0

        self.source = None
        self.dest = None



        self.node_searched = 0
        self.obstacles = []
        self.path = []
        self.children = []

    def update(self, map, source, dest, obstacles = []):
        self.map = map
        self.width = map.width
        self.height = map.height
        self.source = source
        self.dest = dest
        self.node_searched = 0
        self.obstacles = obstacles if len(obstacles) != 0 else []

