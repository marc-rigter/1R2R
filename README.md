# One Risk to Rule Them All: A Risk-Sensitive Perspective on Model-Based Offline Reinforcement Learning	

Official code to reproduce the experiments in the paper ["One Risk to Rule Them All: A Risk-Sensitive Perspective on Model-Based Offline Reinforcement Learning	
"](https://arxiv.org/abs/2212.00124).

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

## Datasets
The datasets introduced for stochastic domains can be found on the [HuggingFace Hub](https://huggingface.co/datasets/marcrigter/1R2R-datasets) as well as [Google Drive](https://arxiv.org/abs/2212.00124](https://drive.google.com/drive/folders/1z52N4MHHlqYRljUT1azIRq2s462ZSQOA?usp=sharing). By default, the code expects that the datasets are located in the folder 1R2R/datasets.

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

## Citing 1R2R
```
@article{rigter2023,
  title={One Risk to Rule Them All: A Risk-Sensitive Perspective on Model-Based Offline Reinforcement Learning},
  author={Rigter, Marc and Lacerda, Bruno and Hawes, Nick},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
