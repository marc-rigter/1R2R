import gym


gym.envs.register(
     id='HIVTreatment-v0',
     entry_point='softlearning.environments.hiv_treatment.hiv_treatment:HIVTreatment',
     max_episode_steps=50,
     kwargs={},
)

gym.envs.register(
     id='Walker2dModNoise-v0',
     entry_point='softlearning.environments.mujoco_stochastic.walker2d_mod_noise:Walker2DModNoise',
     max_episode_steps=1000,
     kwargs={},
)

gym.envs.register(
     id='Walker2dHighNoise-v0',
     entry_point='softlearning.environments.mujoco_stochastic.walker2d_high_noise:Walker2DHighNoise',
     max_episode_steps=1000,
     kwargs={},
)

gym.envs.register(
     id='HopperModNoise-v0',
     entry_point='softlearning.environments.mujoco_stochastic.hopper_mod_noise:HopperModNoise',
     max_episode_steps=1000,
     kwargs={},
)

gym.envs.register(
     id='HopperHighNoise-v0',
     entry_point='softlearning.environments.mujoco_stochastic.hopper_high_noise:HopperHighNoise',
     max_episode_steps=1000,
     kwargs={},
)

gym.envs.register(
     id='CurrencyExchange-v0',
     entry_point='softlearning.environments.currency_exchange.currency_exchange:CurrencyExchange',
     max_episode_steps=50,
     kwargs={},
)