# Drived by modified A* algorithm
import numpy as np
import math
from helper import Coordination


class PathNode:
    def __init__(self, slope_val, map, source, destination, target, parent = None, isStepping = False, obstacles = []):
        self.parent = parent

        self.slope_val = slope_val
        self.map = map

        self.source = source
        self.destination = destination
        self.target = target

        self.open_list = MinPriorList()
        self.close_list = np.zeros((self.map.height, self.map.width), dtype=bool)

        self.obstacles = obstacles if len(obstacles) != 0 else []
        self.path = []
        self.children = []
        self.isStepping = isStepping
        self.reached = False

        self.node_searched = 0

    def start_search(self):
        self.expand()

    def search(self):
        print("SEARCH xhSTARTED!")
        # 将起点加入到open_list中
        start_node = OpenNode(self.source.x, self.source.y, 0, self.calc_pred(self.source), None)
        self.open_list.push(start_node)
        # 执行一次搜索
        self.open_new_node()
        result = start_node
        # 开始循环搜索
        while not self.open_list.is_empty():
            node_to_open = self.open_list.top()
            if node_to_open.get_pos() == (self.destination.x, self.destination.y):
                result = node_to_open
                self.source = result.get_coordination(self.map)
                self.reached = True
                print("SEARCH FINISHED!")
                break
            self.open_new_node()
            if self.node_searched % 100 == 0:
                print(f'Searched {self.node_searched} nodes.')
        print(f'Searched {self.node_searched} nodes.')

        if self.reached:
            path = []
            # 对路径进行溯源
            path.append(result.get_pos())
            path_iterator = PathIterator(result)
            for pos in path_iterator:
                path.append(pos)
    
            self.path = path
            self.obstacles += self.path
            print(self.path)
            # print(f'PATH:{path}')
            print(f'MIN COST:{node_to_open.cost}')
            # 绘制路径的函数
        
    def get_path(self):
        result = []
        results = []
        temp = self.cal_shortest_path()
        while(temp.parent != None):
            results.append(temp.path)
            temp = temp.parent
        for item in results:
            result += item
        return result

    def cal_path(self, leaf):
        result = []
        results = []
        temp_node = leaf
        while(temp_node.parent != None):
            results.append(temp_node.path)
            temp_node = temp_node.parent
        for item in results:
            result += item
        return result

    def cal_shortest_path(self):
        leafs = self.get_all_leaf()
        paths_len = []
        count = 0
        for leaf in leafs:
            count += 1
            print(str(count) + str(self.cal_path(leaf)))
            length = 0
            temp = leaf
            while(temp.parent != None):
                length += len(temp.path)
                temp = temp.parent
            paths_len.append(length) 
        min_len = min(paths_len)
        shortest_leaf = leafs[paths_len.index(min_len)]
        return shortest_leaf

    def get_all_leaf(self):
        if self.children == []:
            return [self]
        else:
            result = []
            for child in self.children:
                result += child.get_all_leaf()
            return result

    def expand(self):
        if self.source.z == self.target.z:
            self.children.append(PathNode(self.slope_val, self.map, self.source, self.target, self.target, parent = self, isStepping = False, obstacles = list(self.obstacles)))        

        # Find the slope
        vals = self.map.dem_map - self.source.z
        temps = np.where(np.absolute(vals) == self.slope_val)
        slope_pts = np.flip(np.transpose(temps), axis = 1)

        count = 0
        for slope_pt in slope_pts.tolist():
            if (slope_pt[0], slope_pt[1]) in self.obstacles:
                continue
            if vals[slope_pt[1]][slope_pt[0]] > 0:
                stop = self.find_slope_destination(self.map.get_coordination(slope_pt[0],slope_pt[1]), True)
            else:
                stop = self.find_slope_destination(self.map.get_coordination(slope_pt[0],slope_pt[1]), False)

            if (stop.x, stop.y) in self.obstacles:
                continue
            count += 1
            print(stop.x, stop.y, stop.z)
            self.children.append(PathNode(self.slope_val, self.map, self.source, stop, self.target, parent = self, isStepping=True, obstacles = list(self.obstacles)))
        print("slope ok for " + str(count))

        dels = []
        for child in self.children:
            child.search()
            if child.reached:
                if child.destination != child.target:
                    child.expand()
            else:
                dels.append(child)
        for del_child in dels:
            self.children.remove(del_child)
    

            

    def calc_cost(self, pos, new_pos):
        if pos.x != new_pos.x and pos.y != new_pos.y:
            horizon_dist = 1.414 * self.map.grid_size
        else:
            horizon_dist = 1 * self.map.grid_size
        cost = math.sqrt(
            horizon_dist ** 2 + (pos.z - new_pos.z) ** 2)
        # print(f'CALC COST. Pos({pos[0]},{pos[1]})->({new_pos[0]},{new_pos[1]}),cost = {cost:.2f}')
        return cost

    def calc_pred(self, location):
        grid_size = self.map.grid_size
        horizon_dist = math.sqrt(
            ((location.x - self.destination.x) * grid_size) ** 2 + ((location.y - self.destination.y) * grid_size) ** 2)
        height_dist = abs(location.z - self.destination.z)
        pred = math.sqrt(horizon_dist ** 2 + height_dist ** 2)
        # print(f'CALC PRED. Pos({location[0]},{location[1]}),pred = {pred:.2f}')
        return pred

    # 遍历节点的周边节点
    def get_new_position(self, location):
    	#对应8个方向,也可以根据需要改为4个方向
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        pos_list = []
        for offset in offsets:
            pos = location.get_coordination(self.map)
            new_pos = Coordination(pos.x + offset[0], pos.y + offset[1])
            #此处判断节点是否超出了地图范围
            if new_pos.x < 0 or new_pos.x >= self.map.width or new_pos.y < 0 or new_pos.y >= self.map.height:
                continue
            new_pos = self.map.get_coordination(new_pos.x, new_pos.y)
            if self.isStepping:
                if abs(pos.z - new_pos.z) > self.slope_val:
                    continue
            else:
                if abs(pos.z - new_pos.z) >= self.slope_val:
                    continue

            if len(self.obstacles) != 0:
                if (new_pos.x, new_pos.y) in self.obstacles:
                    continue
            pos_list.append(new_pos)
        return pos_list

    def open_new_node(self):
        node_to_open = self.open_list.pop()
        pos_list = self.get_new_position(node_to_open)
        for new_pos in pos_list:
            if not self.in_close_list(new_pos):
            	# 判断新位置是否在open_list中
                if self.open_list.in_list(new_pos):
                    # print(f'NODE IN OPEN LIST.{new_pos}')
                    # 父节点->new_node->new_pos的花费
                    grandpa_father_dist = self.calc_cost(node_to_open.get_coordination(self.map), new_pos) + node_to_open.cost
                    # print(f'Direct path:{node_to_open.father.get_pos()}->{node_to_open.get_pos()}->{new_pos},cost={direct_dist:.2f}')
                    # 父节点->new_pos的花费
                    grandpa_dist = self.calc_cost(new_pos, node_to_open.get_coordination(self.map))
                    # print(f'Father path:{node_to_open.father.get_pos()}->{new_pos},cost={father_dist:.2f}')
                    if grandpa_dist <= grandpa_father_dist:
                        # print("UPDATED OPEN NODE!")
                        self.open_list.del_node(new_pos)
                        new_node = OpenNode(new_pos.x, new_pos.y, grandpa_dist, self.calc_pred(new_pos),
                                            node_to_open.father)
                        # new_node.describe_node()
                        self.open_list.push(new_node)
                else:
                    new_node = OpenNode(new_pos.x, new_pos.y, self.calc_cost(node_to_open.get_coordination(self.map), new_pos),
                                        self.calc_pred(new_pos), node_to_open)
                    # new_node.describe_node()
                    self.open_list.push(new_node)
        # 将搜索过的该节点加入到close_list中
        self.close_list[node_to_open.y, node_to_open.x] = True
        self.node_searched += 1
    
    def in_close_list(self, new_pos):
        return self.close_list[new_pos.y, new_pos.x] == True

    def find_slope_destination(self,slope_pt, isStepOn):
        print("SEARCH xhSTARTED! - find_slope_destination")
        # 将起点加入到open_list中
        start_pt = slope_pt
        searched = MinPriorList()
        searched.list.append(start_pt)
        result = start_pt
        # 开始循环搜索
        while not searched.is_empty():
            result = searched.pop()
            for item in self.find_further_slope_pts(result, isStepOn):
                searched.list.append(item)
            print(str(result.x) + " " + str(result.y) + " " + str(result.z))
        return result

    def find_further_slope_pts(self, slope_pt, isStepOn):
            	#对应8个方向,也可以根据需要改为4个方向
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        pos_list = []
        for offset in offsets:
            pos = slope_pt
            new_pos = Coordination(pos.x + offset[0], pos.y + offset[1])
            #此处判断节点是否超出了地图范围
            if new_pos.x < 0 or new_pos.x >= self.map.width or new_pos.y < 0 or new_pos.y >= self.map.height:
                continue
            new_pos = self.map.get_coordination(new_pos.x, new_pos.y)
            if isStepOn:
                if (new_pos.z - pos.z) > self.slope_val or (new_pos.z - pos.z) <= 0:
                    continue
            else:
                if (pos.z - new_pos.z) > self.slope_val or (pos.z - new_pos.z) <= 0:
                    continue
            pos_list.append(new_pos)
        return pos_list



class PathIterator:
    def __init__(self, node):
        self.node = node
        
	# 这里是每一步迭代需要进行的操作
    def __next__(self):
        if self.node.father is not None:
            self.node = self.node.father
            return self.node.get_pos()
        else:
            raise StopIteration
	
	# 在类中添加此方法,就可以进行迭代操作.
    def __iter__(self):
        return self

class OpenNode(Coordination):
    def __init__(self, x, y, cost=0.0, pred=0.0, father=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.pred = pred
        self.father = father
	
	# 用于返回该点的坐标数组
    def get_pos(self):
        return self.x, self.y

    def get_coordination(self, map):
        return map.get_coordination(self.x, self.y)
	
	# 用于得到F(n)的值
    def get_F(self):
        return self.cost + self.pred

class MinPriorList:
    def __init__(self):
        self.list = []

    def is_empty(self):
        return len(self.list) == 0

    def in_list(self, pos):
        for i in range(len(self.list) - 1, -1, -1):
            if self.list[i].get_pos() == (pos.x, pos.y):
                return True
        return False

    def del_node(self, pos):
        for i in range(len(self.list) - 1, -1, -1):
            if self.list[i].get_pos() == (pos.x, pos.y):
                del self.list[i]
                return True
        return False
	
	# 插入一个节点,并移动该节点使得列表按F(n)降序排列
    def push(self, node):
        if self.is_empty():
            self.list.append(node)
        else:
            self.list.append(node)
            for i in range(len(self.list) - 1, 0, -1):
                if self.list[i].get_F() > self.list[i - 1].get_F():
                    self.list[i], self.list[i - 1] = self.list[i - 1], self.list[i]

	# 返回并删除最后一个节点,返回的是F(n)最小的节点
    def pop(self):
        return self.list.pop()
	
	# 查看最后一个节点
    def top(self):
        return self.list[-1]

    def print_list(self):
        print(f'MinPriorList contains {len(self.list)} nodes.')
        for i in range(0, len(self.list)):
            print(f'({self.list[i].get_pos()},{self.list[i].get_F():.2f})', end=' ')
        print()
