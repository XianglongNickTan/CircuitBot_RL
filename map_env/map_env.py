import numpy as np
from gym import spaces, Env
import random


class MapEnv(Env):
    def __init__(self,
                 map_row=8,
                 map_column=8):

        self.row = map_row
        self.column = map_column


        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([self.row - 1, self.column - 1, self.row - 1, self.column - 1, 1]),
            dtype=np.int)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.row, self.column), dtype='float32')


    def reset(self):
        # Reset the state of the environment to an initial state
        map_weight = np.random.randint(1,5,(self.row, self.column))

        map_weight[]


        return self._next_observation()

    def _next_observation(self):
        # Get the data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                                           5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'Volume'].values / MAX_NUM_SHARES,
        ])
        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)
        return obs


    def step(self, action):
        """ Execute one time step within the environment."""
        pass


    def render(self, mode="human"):



