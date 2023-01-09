from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'HopperHighNoise',
    'task': 'v0',
    'exp_name': 'hopper_high_noise_medium'
})
params['kwargs'].update({
    'pool_load_path': 'datasets/hopper-high-noise-medium-v0',
    'dynamic_risk': 'cvar_0.9',  # or 'wang_0.1'
    'rollout_length': 5,
})
