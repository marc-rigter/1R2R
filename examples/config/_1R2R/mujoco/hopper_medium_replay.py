from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'hopper',
    'task': 'medium-replay-v2',
    'exp_name': 'hopper_medium_replay'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/hopper-medium-replay-v2',
    'dynamic_risk': 'cvar_0.5',  # or 'wang_0.75'
    'rollout_length': 1,
})
