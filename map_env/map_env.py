import pybullet as p
import sys
import time
from collections import deque
import cv2
import numpy as np
import pybullet_data
from gym import spaces, Env
from gym.utils import seeding
from map import Map


sys.path.insert(1, "../bullet3/build_cmake/examples/pybullet")
timeStep = 1 / 240.0


class PhysClientWrapper:
    """
    This is used to make sure each BulletRobotEnv has its own physicsClient and
    they do not cross-communicate.
    """

    def __init__(self, other, physics_client_id):
        self.other = other
        self.physicsClientId = physics_client_id

    def __getattr__(self, name):
        if hasattr(self.other, name):
            attr = getattr(self.other, name)
            if callable(attr):
                return lambda *args, **kwargs: self._wrap(attr, args, kwargs)
            return attr
        raise AttributeError(name)

    def _wrap(self, func, args, kwargs):
        kwargs["physicsClientId"] = self.physicsClientId
        return func(*args, **kwargs)



class MapEnv(Env):
    def __init__(self,
                 map_row=8,
                 map_column=8,
                 n_substeps=5,  # Number of simulation steps to do in every env step.
                 done_after=float("inf"),
                 use_gui=False,
                 ):

        self.row = map_row
        self.column = map_column


        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([self.row - 1, self.column - 1, self.row - 1, self.column - 1, 1]),
            dtype=np.int)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.row, self.column), dtype='float32')

        ### pybullet setting ###
        if use_gui:
            physics_client = p.connect(p.GUI)
        else:
            physics_client = p.connect(p.DIRECT)

        self.p = PhysClientWrapper(p, physics_client)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())


        ### initilize map ###
        self.map = Map(self.p)



    def reset(self):
        # Reset the state of the environment to an initial state
        map_weight = np.random.randint(1,5,(self.row, self.column))

        map_weight[]


        return self._next_observation()

    def _next_observation(self):
        # Get the data points for the last 5 days and scale to between 0-1

        return obs


    def step(self, action):
        """ Execute one time step within the environment."""
        pass


    def render(self, mode="human"):



