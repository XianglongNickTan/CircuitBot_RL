class Coordination:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
    

class NodeIterator:
    def __init__(self, node):
        self.node = node
        
    def __next__(self):
        if self.node.parent is not None:
            self.node = self.node.parent
            return self.node.get_pos(), self.node.cost
        else:
            raise StopIteration

    def __iter__(self):
        return self

class Node(Coordination):
    def __init__(self, x, y, z = 0, map = None, cost=0.0, pred=0.0, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.z = 0 if map == None else map.get_z_index(self.x, self.y)
        self.cost = cost
        self.pred = pred
        self.parent = parent
	
    def get_pos(self):
        return self.x, self.y

    def get_coordination(self, map):
        return map.get_coordination(self.x, self.y)
	
    def get_F(self):
        return self.cost + self.pred

class SearchQueue:
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
