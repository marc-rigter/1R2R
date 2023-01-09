from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'Walker2dModNoise',
    'task': 'v0',
    'exp_name': 'walker2d_mod_noise_medium_replay'
})
params['kwargs'].update({
    'pool_load_path': 'datasets/walker2d-mod-noise-medium-replay-v0',
    'dynamic_risk': 'cvar_0.9',  # or 'wang_0.1'
    'rollout_length': 1,
})
