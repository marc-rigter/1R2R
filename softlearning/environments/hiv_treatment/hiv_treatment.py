__credits__ = ["Rushiv Arora"]

import numpy as np

from gym import utils
from gym import spaces
import gym

"""Domain base class"""
from builtins import range
import numpy as np
from scipy.integrate import odeint



class HIVTreatment(gym.Env):

    """
    Simulation of HIV Treatment. The aim is to find an optimal drug schedule.

    **STATE:** The state contains concentrations of 6 different cells:

    * T1: non-infected CD4+ T-lymphocytes [cells / ml]
    * T1*:    infected CD4+ T-lymphocytes [cells / ml]
    * T2: non-infected macrophages [cells / ml]
    * T2*:    infected macrophages [cells / ml]
    * V: number of free HI viruses [copies / ml]
    * E: number of cytotoxic T-lymphocytes [cells / ml]

    **ACTIONS:** The therapy consists of 2 drugs
    (reverse transcriptase inhibitor [RTI] and protease inhibitor [PI]) which
    are activated or not. The action space contains therefore of 4 actions:

    * *0*: none active
    * *1*: RTI active
    * *2*: PI active
    * *3*: RTI and PI active

    **REFERENCE:**

    .. seealso::
        Ernst, D., Stan, G., Gonc, J. & Wehenkel, L.
        Clinical data based optimal STI strategies for HIV:
        A reinforcement learning approach
        In Proceedings of the 45th IEEE Conference on Decision and Control (2006).


    """


    def __init__(self):
        self.obs_low = np.array(np.ones(6) * -5)
        self.obs_high = np.array(np.ones(6) * 8)
        act_low = np.array(np.ones(2) * -1)
        act_high = np.array(np.ones(2))
        obs_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.observation_space = spaces.Dict({"observation": obs_space})
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        self.active_observation_shape = obs_space.shape

        self.num_steps = 50
        self.dt = 20  #: measurement every 20 days
        self.logspace = True  #: whether observed states are in log10 space or not

        self.dosage_noise = 0.15
        self.max_noise = 0.3
        self.max_eps1 = 0.7
        self.max_eps2 = 0.3

    def seed(self, seed):
        np.random.seed(seed)


    def step(self, a):
        self.t += 1
        # if self.logspace:
        #    s = np.power(10, s)

        eps1, eps2 = a[0], a[1]

        # rescale to action space
        eps1 = (eps1 + 1) / 2 * self.max_eps1
        eps2 = (eps2 + 1) / 2 * self.max_eps1

        # scale by noise level
        eps1 =  eps1 * (1 + np.random.normal(scale=self.dosage_noise))
        eps2 =  eps2 * (1 + np.random.normal(scale=self.dosage_noise))

        # clip
        eps1 = np.clip(eps1, 0.0, (1 + self.max_noise) * self.max_eps1)
        eps2 = np.clip(eps2, 0.0, (1 + self.max_noise) * self.max_eps2)

        ns = odeint(dsdt, self.state, [0, self.dt],
                    args=(eps1, eps2), mxstep=1000)[-1]
        T1, T2, T1s, T2s, V, E = ns
        # the reward function penalizes treatment because of side-effects
        reward = - 0.1 * V - 2e4 * eps1 ** 2 - 2e3 * eps2 ** 2 + 1e3 * E
        reward = reward / 1e6 - 1.0
        self.state = ns.copy()
        if self.logspace:
            ns = np.log10(ns)

        terminal = False
        if self.t == self.num_steps:
            terminal = True

        return ns, reward, terminal, {}


    def reset(self):
        self.t = 0
        # non-healthy stable state of the system
        s = np.array([163573., 5., 11945., 46., 63919., 24.])
        self.state = s.copy()

        if self.logspace:
            return np.log10(s)

        return s



def dsdt(s, t, eps1, eps2):
    """
    system derivate per time. The unit of time are days.
    """
    # model parameter constants
    lambda1 = 1e4
    lambda2 = 31.98
    d1 = 0.01
    d2 = 0.01
    f = .34
    k1 = 8e-7
    k2 = 1e-4
    delta = .7
    m1 = 1e-5
    m2 = 1e-5
    NT = 100.
    c = 13.
    rho1 = 1.
    rho2 = 1.
    lambdaE = 1
    bE = 0.3
    Kb = 100
    d_E = 0.25
    Kd = 500
    deltaE = 0.1

    # decompose state
    T1, T2, T1s, T2s, V, E = s

    # compute derivatives
    tmp1 = (1. - eps1) * k1 * V * T1
    tmp2 = (1. - f * eps1) * k2 * V * T2
    dT1 = lambda1 - d1 * T1 - tmp1
    dT2 = lambda2 - d2 * T2 - tmp2
    dT1s = tmp1 - delta * T1s - m1 * E * T1s
    dT2s = tmp2 - delta * T2s - m2 * E * T2s
    dV = (1. - eps2) * NT * delta * (T1s + T2s) - c * V \
        - ((1. - eps1) * rho1 * k1 * T1 +
           (1. - f * eps1) * rho2 * k2 * T2) * V
    dE = lambdaE + bE * (T1s + T2s) / (T1s + T2s + Kb) * E \
        - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E - deltaE * E

    return np.array([dT1, dT2, dT1s, dT2s, dV, dE])
