import numpy as np
import math
from utils.pathplanning.helper import Coordination
from utils.pathplanning.helper import Node
from utils.pathplanning.helper import SearchQueue
from utils.pathplanning.helper import NodeIterator

class PathPlanner:
    def __init__(self, analyzer):
        self.analyzer = analyzer

        self.departure = None
        self.destination = None

        self.cost = 0

        self.open_list = None
        self.close_list = None
        self.node_searched = 0

        self.count = 0

        self.path = []
        self.reached = False
        
        self.obstacles = []
        self.result_path = []

    def get_map(self):
        return self.analyzer.map

    def set_pathplan(self, departure, destination):
        self.departure = departure
        self.destination = destination

    def add_obstacles(self, obstacles):
        self.obstacles += obstacles

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def clear_obstacles(self):
        self.obstacles.clear()

    def search(self):
        self.open_list = SearchQueue()
        self.close_list = np.zeros((self.get_map().height, self.get_map().width), dtype=bool)
        self.node_searched = 0
        
        start_node = Node(self.departure.x, self.departure.y, self.departure.z, self.get_map(), 0, self.calc_pred(self.departure), None)
        self.open_list.push(start_node)
        
        self.expand()
        result = start_node

        count = 0
        
        while not self.open_list.is_empty():
            node_to_open = self.open_list.top()
            if node_to_open.get_pos() == (self.destination.x, self.destination.y):
                print(self.count)
                result = node_to_open
                path = []
                path.append(result.get_pos())
                total_cost = result.cost
                node_iterator = NodeIterator(result)
                for pos, cost in node_iterator:
                    path.append(pos)
                    total_cost += cost
                self.path = path
                self.cost = total_cost
                return PathResult(True, path, total_cost)
            count += 1
            self.expand()

        return PathResult()
    
    def calc_cost(self, pos, new_pos):
        if pos.x != new_pos.x and pos.y != new_pos.y:
            horizon_dist = 1.414 * self.get_map().grid_size
        else:
            horizon_dist = 1 * self.get_map().grid_size
        cost = math.sqrt(
            horizon_dist ** 2 + (pos.z - new_pos.z) ** 2)
        return cost

    def calc_pred(self, location):
        grid_size = self.get_map().grid_size
        horizon_dist = math.sqrt(
            ((location.x - self.destination.x) * grid_size) ** 2 + ((location.y - self.destination.y) * grid_size) ** 2)
        height_dist = abs(location.z - self.destination.z)
        pred = math.sqrt(horizon_dist ** 2 + height_dist ** 2)
        return pred

    def get_new_position(self, location):
        # offsets = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        pos_list = []
        for offset in offsets:
            pos = location.get_coordination(self.get_map())
            new_pos = Coordination(pos.x + offset[0], pos.y + offset[1])
            if new_pos.x < 0 or new_pos.x >= self.get_map().width or new_pos.y < 0 or new_pos.y >= self.get_map().height:
                continue
            new_pos = self.get_map().get_coordination(new_pos.x, new_pos.y)

            if abs(pos.z - new_pos.z) > self.analyzer.max_slope:
                continue

            if len(self.obstacles) != 0:
                if (new_pos.x, new_pos.y) in self.obstacles:
                    return []
            pos_list.append(new_pos)
        return pos_list

    def expand(self):
        node_to_open = self.open_list.pop()
        pos_list = self.get_new_position(node_to_open)
        for new_pos in pos_list:
            if not self.in_close_list(new_pos):
                if self.open_list.in_list(new_pos):
                    cost_from_current_node = self.calc_cost(node_to_open.get_coordination(self.get_map()), new_pos) + node_to_open.cost
                    cost_from_parent_node  = self.calc_cost(new_pos, node_to_open.parent.get_coordination(self.get_map()))
                    if cost_from_parent_node <= cost_from_current_node:
                        self.open_list.del_node(new_pos)
                        new_node = Node(new_pos.x, new_pos.y, new_pos.z, self.get_map(), cost_from_parent_node, self.calc_pred(new_pos),
                                            node_to_open.parent)
                        self.open_list.push(new_node)
                else:
                    new_node = Node(new_pos.x, new_pos.y, new_pos.z, self.get_map(), self.calc_cost(node_to_open.get_coordination(self.get_map()), new_pos),
                                        self.calc_pred(new_pos), node_to_open)
                    self.open_list.push(new_node)

        self.close_list[node_to_open.y, node_to_open.x] = True
        self.node_searched += 1
    
    def in_close_list(self, new_pos):
        return self.close_list[new_pos.y, new_pos.x] == True

class PathResult:
    def __init__(self, success = False, path = [], cost = -1):
        self.success = success
        self.path = path
        self.cost = cost

    def __str__(self):
        return str(self.success) + str(self.path) + str(self.cost)

class PathIterator(NodeIterator):
    def __init__(self, node):
        self.node = node
        
    def __next__(self):
        if self.node.parent is not None:
            self.node = self.node.parent
            return self.node.get_pos(), self.node.cost, self.node.path
        else:
            raise StopIteration

    def __iter__(self):
        return self