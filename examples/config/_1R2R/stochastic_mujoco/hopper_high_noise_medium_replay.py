from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'HopperHighNoise',
    'task': 'v0',
    'exp_name': 'hopper_high_noise_medium_replay'
})
params['kwargs'].update({
    'pool_load_path': 'datasets/hopper-high-noise-medium-replay-v0',
    'dynamic_risk': 'cvar_0.5',  # or 'wang_0.75'
    'rollout_length': 5,
})
