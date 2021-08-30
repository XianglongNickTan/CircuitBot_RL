import math

class BFSSearch():
    def update(self, map, source, dest, obstacles = []):
        self.map = map
        self.width = map.width
        self.height = map.height
        self.source = source
        self.dest = dest

        self.node_searched = 0
        self.obstacles = obstacles if len(obstacles) != 0 else []

    def search(self):
        pass
    
    def expand(self):
        if self.source.z == self.target.z:
            self.children.append(DestinationNode())
        self.children.append(LevelNode())
        for child in self.children:
            child.search()

    # 遍历节点的周边节点
    def get_new_position(self, location):
        pass