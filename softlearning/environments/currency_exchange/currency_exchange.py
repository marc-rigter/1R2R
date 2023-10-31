__credits__ = ["Rushiv Arora"]

import numpy as np

from gym import utils
from gym import spaces
import gym

"""Domain base class"""
from builtins import range
import numpy as np
from scipy.integrate import odeint



class CurrencyExchange(gym.Env):

    """
    Currency exchange domain.

    There are three state features:
    s[0]: t, the time step in {0, 1, 2, ..., 50}
    s[1]: m, the amount of money remaining to exchange, [0, 100]
    s[2]: p, the exchange rate, [0, 3]

    The action represents
    """


    def __init__(self):
        self.obs_low = np.array(np.zeros(3))
        self.obs_high = np.array([50, 100, 5])
        act_low = np.array(np.ones(1) * -1)
        act_high = np.array(np.ones(1))
        obs_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.observation_space = spaces.Dict({"observation": obs_space})
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        self.active_observation_shape = obs_space.shape

        # parameters for price model
        self.price_mu = 1.5
        self.price_sigma = 0.2
        self.price_theta = 0.05

        # initial price
        self.init_price_mu = 1.0
        self.init_price_sigma = 0.05

        self.num_steps = 20
        self.dt = 1
        self.state = self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, a):
        t = self.state[0]
        m = self.state[1]
        p = self.state[2]

        t_next = t + 1
        m_next = m * (1 - np.clip(a, 0, 1))
        reward = ((m - m_next) * p).item()
        p_next = p + self.price_theta * (self.price_mu - p) + self.price_sigma * np.random.normal() * np.sqrt(self.dt)
        p_next = np.clip(p_next, 0, 5)

        if int(np.round(t_next)) == self.num_steps or (m_next < 0.1):
            terminal = True
        else:
            terminal = False

        s_next = np.array([t_next, m_next.item(), p_next.item()])
        self.state = s_next.copy()
        return s_next, reward, terminal, {}

    def reset(self):
        t = 0
        m = 100
        p = np.random.normal(loc=self.init_price_mu, scale=self.init_price_sigma)
        s = np.array([t, m, p])
        self.state = s.copy()
        return s