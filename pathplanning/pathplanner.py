import numpy as np
import math

# F(n)=g(n)+h(n)
class PathPlanner:
    def __init__(self):
        self.map = None
        self.source = None
        self.open_list = MinPriorList()
        self.width = 0
        self.height = 0
        self.dest = [1.0]
        self.close_list = None
        self.node_searched = 0
        self.path = []

    def update(self, map, source, dest):
        self.map = map
        self.width = map.width
        self.height = map.height
        self.source = source
        self.dest = dest
        self.open_list = MinPriorList()
        self.close_list = np.zeros((self.height, self.width), dtype=bool)
        self.node_searched = 0

    def start_search(self):
        print("SEARCH xhSTARTED!")
        # 将起点加入到open_list中
        start_node = OpenNode(self.source[0], self.source[1], 0, self.calc_pred(self.source), None)
        self.open_list.push(start_node)
        # 执行一次搜索
        self.open_new_node()
        # 开始循环搜索
        while not self.open_list.is_empty():
            node_to_open = self.open_list.top()
            if node_to_open.get_pos() == self.dest:
                print("SEARCH FINISHED!")
                break
            self.open_new_node()
            if self.node_searched % 100 == 0:
                print(f'Searched {self.node_searched} nodes.')
        path = []
        # 对路径进行溯源
        path_iterator = PathIterator(node_to_open)
        start_to_append = False
        for pos in path_iterator:
            if pos[0] == self.dest[0] and pos[1] == self.dest[1]:
                start_to_append = True
            if start_to_append:
                path.append(pos)

        self.path = path
        print(self.path)
        # print(f'PATH:{path}')
        print(f'MIN COST:{node_to_open.cost}')
        # 绘制路径的函数
        
    def calc_cost(self, pos, new_pos):
        if pos[0] != new_pos[0] and pos[1] != new_pos[1]:
            horizon_dist = 1.414 * self.map.grid_size
        else:
            horizon_dist = 1 * self.map.grid_size
        cost = math.sqrt(
            horizon_dist ** 2 + (self.map.dem_map[pos[1], pos[0]] - self.map.dem_map[new_pos[1], new_pos[0]]) ** 2)
        # print(f'CALC COST. Pos({pos[0]},{pos[1]})->({new_pos[0]},{new_pos[1]}),cost = {cost:.2f}')
        return cost

    def calc_pred(self, location):
        grid_size = self.map.grid_size
        horizon_dist = math.sqrt(
            ((location[0] - self.dest[0]) * grid_size) ** 2 + ((location[1] - self.dest[1]) * grid_size) ** 2)
        height_dist = abs(self.map.dem_map[location[1], location[0]] - self.map.dem_map[self.dest[1], self.dest[0]])
        pred = math.sqrt(horizon_dist ** 2 + height_dist ** 2)
        # print(f'CALC PRED. Pos({location[0]},{location[1]}),pred = {pred:.2f}')
        return pred

    # 遍历节点的周边节点
    def get_new_position(self, location):
    	#对应8个方向,也可以根据需要改为4个方向
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        pos_list = []
        for offset in offsets:
            pos = location.get_pos()
            new_pos = (pos[0] + offset[0], pos[1] + offset[1])
            #此处判断节点是否超出了地图范围
            if new_pos[0] < 0 or new_pos[0] >= self.width or new_pos[1] < 0 or new_pos[1] >= self.height or abs(self.map.dem_map[pos[1],pos[0]] - self.map.dem_map[new_pos[1],new_pos[0]]) > 0.5:
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
                    grandpa_father_dist = self.calc_cost(node_to_open.get_pos(), new_pos) + node_to_open.cost
                    # print(f'Direct path:{node_to_open.father.get_pos()}->{node_to_open.get_pos()}->{new_pos},cost={direct_dist:.2f}')
                    # 父节点->new_pos的花费
                    grandpa_dist = self.calc_cost(new_pos, node_to_open.father.get_pos())
                    # print(f'Father path:{node_to_open.father.get_pos()}->{new_pos},cost={father_dist:.2f}')
                    if grandpa_dist <= grandpa_father_dist:
                        # print("UPDATED OPEN NODE!")
                        self.open_list.del_node(new_pos)
                        new_node = OpenNode(new_pos[0], new_pos[1], grandpa_dist, self.calc_pred(new_pos),
                                            node_to_open.father)
                        # new_node.describe_node()
                        self.open_list.push(new_node)
                else:
                    new_node = OpenNode(new_pos[0], new_pos[1], self.calc_cost(node_to_open.get_pos(), new_pos),
                                        self.calc_pred(new_pos), node_to_open)
                    # new_node.describe_node()
                    self.open_list.push(new_node)
        # 将搜索过的该节点加入到close_list中
        self.close_list[node_to_open.y, node_to_open.x] = True
        self.node_searched += 1
    
    def in_close_list(self, new_pos):
        return self.close_list[new_pos[1], new_pos[0]] == True
    
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


class OpenNode:
    def __init__(self, x, y, cost=0.0, pred=0.0, father=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.pred = pred
        self.father = father
	
	# 用于返回该点的坐标数组
    def get_pos(self):
        return self.x, self.y
	
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
            if self.list[i].get_pos() == pos:
                return True
        return False

    def del_node(self, pos):
        for i in range(len(self.list) - 1, -1, -1):
            if self.list[i].get_pos() == pos:
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

