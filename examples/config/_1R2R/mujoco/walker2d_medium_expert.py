from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'walker2d',
    'task': 'medium-expert-v2',
    'exp_name': 'walker2d_medium_expert'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/walker2d-medium-expert-v2',
    'dynamic_risk': 'cvar_0.7',  # or 'wang_0.5'
    'rollout_length': 1,
})
