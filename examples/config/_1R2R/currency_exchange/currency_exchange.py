from ..base import base_params
from copy import deepcopy

params = deepcopy(base_params)
params.update({
    'domain': 'CurrencyExchange',
    'task': 'v0',
    'exp_name': 'currency_exchange_random'
})
params['kwargs'].update({
    'pool_load_path': 'datasets/currencyexchange-random-v0',
    'normalize_rewards': True,

    'dynamic_risk': 'cvar_0.5',
    'rollout_length': 1,
    'num_elites': 1,
    'n_epochs': 200,
    'eval_n_episodes': 100,
})
