import numpy as np
import math
from helper import Coordination
from helper import Node
from helper import SearchQueue
from helper import NodeIterator

class PathNode(Node):
    def __init__(self, x, y, z, path_planner, cost = 0, pred = 0, path = [], isStepping = True, parent = None):
        self.path_planner = path_planner
        self.parent = parent

        self.x = x
        self.y = y
        self.z = z

        self.close_list = path_planner.close_list
        
        self.cost = cost
        self.pred = pred

        self.path = path
        self.obstacles = self.path_planner.obstacles if len(self.path_planner.obstacles) != 0 else []

        self.isStepping = isStepping
        self.reached = False

        self.node_searched = 0

    def get_map(self):
        return self.path_planner.get_map()

    def calc_path(self, start, stop):
        open_list = SearchQueue()
        reached = False

        # 将起点加入到open_list中
        start_node = Node(start.x, start.y, start.z, self.get_map(), 0, self.calc_pred(start), None)
        open_list.push(start_node)
        # 执行一次搜索
        self.expand(open_list)
        result = start_node
        # 开始循环搜索
        while not open_list.is_empty():
            node_to_open = open_list.top()
            if node_to_open.get_pos() == (stop.x, stop.y):
                result = node_to_open
                reached = True
                break
            self.expand(open_list)
            # print(node_to_open.parent.x, node_to_open.parent.y)

        total_cost = 0
        if reached:
            path = []
            # 对路径进行溯源
            path.append(result.get_pos())
            node_iterator = NodeIterator(result)
            for pos, cost in node_iterator:
                path.append(pos)
                total_cost += cost
            return NodeResult(True, path, total_cost)
        return NodeResult(False)
    
    def calc_cost(self, pos, new_pos):
        if pos.x != new_pos.x and pos.y != new_pos.y:
            horizon_dist = 1.414 * self.get_map().grid_size
        else:
            horizon_dist = 1 * self.get_map().grid_size
        cost = math.sqrt(
            horizon_dist ** 2 + (pos.z - new_pos.z) ** 2)
        # print(f'CALC COST. Pos({pos[0]},{pos[1]})->({new_pos[0]},{new_pos[1]}),cost = {cost:.2f}')
        return cost

    def calc_pred(self, location):
        grid_size = self.get_map().grid_size
        horizon_dist = math.sqrt(
            ((location.x - self.path_planner.destination.x) * grid_size) ** 2 + ((location.y - self.path_planner.destination.y) * grid_size) ** 2)
        height_dist = abs(location.z - self.path_planner.destination.z)
        pred = math.sqrt(horizon_dist ** 2 + height_dist ** 2)
        # print(f'CALC PRED. Pos({location[0]},{location[1]}),pred = {pred:.2f}')
        return pred

    # 遍历节点的周边节点
    def get_new_position(self, location):
    	#对应8个方向,也可以根据需要改为4个方向
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        pos_list = []
        for offset in offsets:
            pos = location.get_coordination(self.get_map())
            new_pos = Coordination(pos.x + offset[0], pos.y + offset[1])
            #此处判断节点是否超出了地图范围
            if new_pos.x < 0 or new_pos.x >= self.get_map().width or new_pos.y < 0 or new_pos.y >= self.get_map().height:
                continue
            new_pos = self.get_map().get_coordination(new_pos.x, new_pos.y)

            if self.isStepping:
                if abs(pos.z - new_pos.z) > self.path_planner.analyzer.max_slope:
                    continue
            else:
                if abs(pos.z - new_pos.z) >= self.path_planner.analyzer.max_slope:
                    continue

            if len(self.obstacles) != 0:
                if (new_pos.x, new_pos.y) in self.obstacles:
                    continue
            pos_list.append(new_pos)
        return pos_list

    def expand(self, open_list):
        node_to_open = open_list.pop()
        # print(node_to_open.x, node_to_open.y)
        pos_list = self.get_new_position(node_to_open)
        for new_pos in pos_list:
            if not self.in_close_list(new_pos):
                if open_list.in_list(new_pos):
                    cost_from_current_node = self.calc_cost(node_to_open.get_coordination(self.get_map()), new_pos) + node_to_open.cost
                    cost_from_parent_node  = self.calc_cost(new_pos, node_to_open.parent.get_coordination(self.get_map()))
                    if cost_from_parent_node <= cost_from_current_node:
                        # print("UPDATED OPEN NODE!")
                        open_list.del_node(new_pos)
                        new_node = Node(new_pos.x, new_pos.y, new_pos.z, self.get_map(), cost_from_parent_node, self.calc_pred(new_pos),
                                            node_to_open.parent)
                        # new_node.describe_node()
                        open_list.push(new_node)
                else:
                    new_node = Node(new_pos.x, new_pos.y, new_pos.z, self.get_map(), self.calc_cost(node_to_open.get_coordination(self.get_map()), new_pos),
                                        self.calc_pred(new_pos), node_to_open)
                    # new_node.describe_node()
                    open_list.push(new_node)
        # 将搜索过的该节点加入到close_list中
        self.close_list[node_to_open.y, node_to_open.x] = True
        self.node_searched += 1
    
    def in_close_list(self, new_pos):
        return self.close_list[new_pos.y, new_pos.x] == True

class NodeResult():
    def __init__(self, success, path = [], cost = -1):
        self.success = success
        self.path = path
        self.cost = cost

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