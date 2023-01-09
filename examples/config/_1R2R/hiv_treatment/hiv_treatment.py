from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'HIVTreatment',
    'task': 'v0',
    'exp_name': 'hiv_treatment_medium_replay'
})
params['kwargs'].update({
    'pool_load_path': 'datasets/hivtreatment-medium-replay-v0',
    'normalize_rewards': True,

    'dynamic_risk': 'cvar_0.7', # or 'wang_0.5'
    'rollout_length': 1,
})
