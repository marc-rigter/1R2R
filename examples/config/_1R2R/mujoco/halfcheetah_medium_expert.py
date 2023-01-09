from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'halfcheetah',
    'task': 'medium-expert-v2',
    'exp_name': 'halfcheetah_medium_expert'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/halfcheetah-medium-expert-v2',
    'dynamic_risk': 'cvar_0.9',  # or 'wang_0.1'
    'rollout_length': 5,
})
