from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'Walker2dHighNoise',
    'task': 'v0',
    'exp_name': 'walker2d_high_noise_medium'
})
params['kwargs'].update({
    'pool_load_path': 'datasets/walker2d-high-noise-medium-v0',
    'dynamic_risk': 'cvar_0.7',  # or 'wang_0.1'
    'rollout_length': 1,
})
