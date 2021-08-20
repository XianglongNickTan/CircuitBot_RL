import numpy as np
from gym import spaces, Env
import random


class Map():
    def __init__(self,
                 map_row=8,
                 map_column=8):

        self.row = map_row
        self.column = map_column

        self.weight_map = np.zeros((self.row, self.column), dtype="float32")

        self.pre_map = self.weight_map

        self.path_length = 0

        self.success = False

        self.shape_rec = np.ones((1, 3), dtype="float32")

        self.init_map()





    def init_map(self, rgb_map=None, depth_map=None):
        """ generate init map from rgb and depth map."""

        self.weight_map[2, 2:5] = self.shape_rec




    def apply_action(self, action):
        """ apply action to update the map.
            action = [x, y]
        """
        move_point = action

        action = np.clip(action, [0, 1], [7, 6])

        self.weight_map = np.zeros((self.row, self.column), dtype="float32")

        self.weight_map[action[0], action[1] - 1: action[1] + 2] = self.shape_rec







    #
    # def get_observation(self):
    #     """ return observation. """
    #
    #     pass

