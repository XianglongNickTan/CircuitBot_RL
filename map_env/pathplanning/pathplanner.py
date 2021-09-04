import numpy as np
import math
from pathnode import PathNode
from pathnode import PathIterator
from helper import Coordination
from helper import Node
from helper import SearchQueue
from helper import NodeIterator

class PathPlanner:
    def __init__(self, analyzer):
        self.analyzer = analyzer

        self.departure = None
        self.destination = None

        self.cost = 0

        self.open_list = None
        self.close_list = None
        self.node_searched = 0

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

    def clear_obstacles(self, obstacles):
        self.obstacles.clear()

    def search(self):
        self.open_list = SearchQueue()
        self.close_list = np.zeros((self.get_map().height, self.get_map().width), dtype=bool)
        self.node_searched = 0

        start_node = PathNode(self.departure.x, self.departure.y, self.departure.z, self)
        self.open_list.push(start_node)
        self.expand()
        result = start_node
        while not self.open_list.is_empty():
            node_to_open = self.open_list.top()
            if node_to_open == self.destination:
                result = node_to_open
                self.reached = True
                # print("SEARCH FINISHED!")
                break
            self.expand()
            if self.node_searched % 100 == 0:
                pass
                # print(f'Searched {self.node_searched} nodes.')
        # print(f'Searched {self.node_searched} nodes.')

        if self.reached:
            resultNodes = []
            path = []
            total_cost = 0
            # 对路径进行溯源
            resultNodes.append(result.get_pos())
            self.resultnode = result
            path += result.path
            total_cost += result.cost
            path_iterator = PathIterator(result)
            for pos, cost, _path in path_iterator:
                resultNodes.append(pos)
                path += _path
                total_cost += cost
    
            self.path = path
            self.cost = total_cost
            self.pathnodes = resultNodes
            
            return True
        else:
            return False
        self.node_searched = 0

    def expand(self):
        node_to_open = self.open_list.pop()
        node_list = self.get_new_nodes(node_to_open)
        for new_node, isStepping in node_list:
            if not self.in_close_list(new_node):
                if self.open_list.in_list(new_node):
                    path_from_current_node_result = node_to_open.calc_path(node_to_open, new_node) 
                    path_from_parent_node_result  = node_to_open.calc_path(node_to_open.parent, new_node)
                    if path_from_current_node_result.success and path_from_parent_node_result.success:
                        if path_from_parent_node_result.cost <= (path_from_current_node_result.cost + node_to_open.cost):
                            self.open_list.del_node(new_node)
                            new_node = PathNode(new_node.x, new_node.y, new_node.z, self,
                                                cost = path_from_parent_node_result.cost, pred = self.calc_pred(new_node),
                                                path = path_from_parent_node_result.path, 
                                                isStepping = isStepping, parent = node_to_open.parent)
                            self.open_list.push(new_node)
                else:
                    path_from_current_node_result = node_to_open.calc_path(node_to_open, new_node)
                    if path_from_current_node_result.success:
                        new_node = PathNode(new_node.x, new_node.y, new_node.z, self,
                                            cost = path_from_current_node_result.cost,
                                            pred = self.calc_pred(new_node), path = path_from_current_node_result.path, 
                                            isStepping = isStepping, parent = node_to_open)
                        self.open_list.push(new_node)
        # 将搜索过的该节点加入到close_list中
        self.close_list[node_to_open.y, node_to_open.x] = True
        self.node_searched += 1

    def in_close_list(self, new_pos):
        return self.close_list[new_pos.y, new_pos.x] == True

    def get_new_nodes(self, location):
        pos_list = []

        if location.z == self.destination.z:
            pos_list.append((self.get_map().get_coordination(self.destination.x, self.destination.y), False))

        vals = self.get_map().dem_map - location.z
        temps = np.where(self.analyzer.min_slope < np.absolute(vals))
        temps = np.where(np.absolute(temps) < self.analyzer.max_slope)
        slope_pts = np.flip(np.transpose(temps), axis = 1)

        # for slope_pt in slope_pts.tolist():
        #     if (slope_pt[0], slope_pt[1]) in self.obstacles:
        #         continue
        #     pos_list.append((self.get_map().get_coordination(slope_pt[0], slope_pt[1]), True))

        count = 0
        for slope_pt in slope_pts.tolist():
            if (slope_pt[0], slope_pt[1]) in self.obstacles:
                continue
            if vals[slope_pt[1]][slope_pt[0]] > 0:
                stop = self.find_slope_destination(self.get_map().get_coordination(slope_pt[0],slope_pt[1]), True)
            else:
                stop = self.find_slope_destination(self.get_map().get_coordination(slope_pt[0],slope_pt[1]), False)

            if (stop.x, stop.y) in self.obstacles:
                continue
            count += 1
            pos_list.append((self.get_map().get_coordination(stop.x, stop.y), True))
        # print("slope ok for " + str(count))
        return pos_list
    
    def find_slope_destination(self, slope_pt, isStepOn):
        # print("SEARCH xhSTARTED! - find_slope_destination")
        # 将起点加入到open_list中
        start_pt = slope_pt
        searched = SearchQueue()
        searched.list.append(start_pt)
        result = start_pt
        # 开始循环搜索
        while not searched.is_empty():
            result = searched.pop()
            for item in self.find_further_slope_pts(result, isStepOn):
                searched.list.append(item)
            # print(str(result.x) + " " + str(result.y) + " " + str(result.z))
        return result

    def find_further_slope_pts(self, slope_pt, isStepOn):
            	#对应8个方向,也可以根据需要改为4个方向
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        pos_list = []
        for offset in offsets:
            pos = slope_pt
            new_pos = Coordination(pos.x + offset[0], pos.y + offset[1])
            #此处判断节点是否超出了地图范围
            if new_pos.x < 0 or new_pos.x >= self.get_map().width or new_pos.y < 0 or new_pos.y >= self.get_map().height:
                continue
            new_pos = self.get_map().get_coordination(new_pos.x, new_pos.y)
            if isStepOn:
                if (new_pos.z - pos.z) > self.analyzer.max_slope or (new_pos.z - pos.z) <= 0:
                    continue
            else:
                if (pos.z - new_pos.z) > self.analyzer.max_slope or (pos.z - new_pos.z) <= 0:
                    continue
            pos_list.append(new_pos)
        return pos_list
    
    def calc_cost(self, node):
        pass
    
    def calc_pred(self, node):
        grid_size = self.get_map().grid_size
        horizon_dist = math.sqrt(
            ((node.x - self.destination.x) * grid_size) ** 2 + ((node.y - self.destination.y) * grid_size) ** 2)
        height_dist = abs(node.z - self.destination.z)
        pred = math.sqrt(horizon_dist ** 2 + height_dist ** 2)
        # print(f'CALC PRED. Pos({location[0]},{location[1]}),pred = {pred:.2f}')
        return pred
    



