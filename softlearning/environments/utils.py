from .adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter,
)
from .mdp_examples.illustrative_example import IllustrativeExample
from .hiv_treatment.hiv_treatment import HIVTreatment
import rambo.env as env_overwrite
import pdb

ENVIRONMENTS = {
    'gym': GYM_ENVIRONMENTS,
}

ADAPTERS = {
    'gym': GymAdapter,
}


def get_environment(universe, domain, task, environment_params):

    if "gym" not in universe:
        return get_mdp_example(domain)

    if domain in env_overwrite:
        print('[ environments/utils ] WARNING: Using overwritten {} environment'.format(domain))
        env = env_overwrite[domain]()
        env = ADAPTERS[universe](None, None, env=env)
    else:
        env = ADAPTERS[universe](domain, task, **environment_params)
    return env

def get_environment_from_params(environment_params):
    universe = environment_params['universe']
    task = environment_params['task']
    domain = environment_params['domain']
    environment_kwargs = environment_params.get('kwargs', {}).copy()

    return get_environment(universe, domain, task, environment_kwargs)

def get_mdp_example(domain):
    if domain == "IllustrativeExample":
        env = IllustrativeExample()
    elif domain == "HIVTreatment":
        env = HIVTreatment()
    env = GymAdapter(None, None, env=env)
    return env
