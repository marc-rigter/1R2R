# One Risk to Rule Them All: Addressing Distributional Shift in Offline Reinforcement Learning via Risk-Aversion 	

Official code to reproduce the experiments in the [1R2R paper](https://arxiv.org/abs/2212.00124).

## Installation
1. Install [MuJoCo 2.1.0](https://github.com/deepmind/mujoco/releases) to `~/.mujoco/mujoco210`.
2. Create a conda environment and install 1R2R:
```
cd 1R2R
conda create --name 1R2R python=3.7
conda activate 1R2R
pip install -e .
pip install -r requirements.txt
```

## Usage
Configuration files can be found in `examples/config/`. For example, to run the stochastic  hopper-medium-replay task with high noise, use the following:

```
1R2R run_example examples.development --config examples.config._1R2R.stochastic_mujoco.hopper_high_noise_medium_replay --seed 0 --gpus 1
```

If importlib is unable to import the desired config file, this can be resolved by adding to the PYTHONPATH:
```
export PYTHONPATH="${PYTHONPATH}:/path/to/1R2R"
```

#### Logging

By default, TensorBoard logs are generated in the "logs" directory. The code is also set up to log using Weights and Biases (WandB). To enable the use of WandB, set "log_wandb" to True in the configuration file.
