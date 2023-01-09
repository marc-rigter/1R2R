from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'HopperModNoise',
    'task': 'v0',
    'exp_name': 'hopper_mod_noise_medium_replay'
})
params['kwargs'].update({
    'pool_load_path': 'datasets/hopper-mod-noise-medium-replay-v0',
    'dynamic_risk': 'cvar_0.7',  # or 'wang_0.5'
    'rollout_length': 5,
})
