## Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient

This repository provides an implementation of [Drama](https://openreview.net/forum?id=7XIkRgYjK3&nesting=2&sort=date-desc): a Mamba/Mamba2 powered model based reinforcement learning agent.

If you find Drama useful, please reference in your paper:
```
@inproceedings{
	wang2025drama,
	title={Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient},
	author={Wenlong Wang and Ivana dusparic and Yucheng Shi and Ke Zhang and Vinny Cahill},
	booktitle={The Thirteenth International Conference on Learning Representations},
	year={2025},
	url={https://openreview.net/forum?id=7XIkRgYjK3}
}
```

## Training and Evaluating Instructions
### Requirements

- **Base Container (Recommended)**: `nvcr.io/nvidia/pytorch:24.05-py3`
- **Python**: 3.10 (included in the base container)
- **CUDA**: 12.4.1 (included in the base container)
- **PyTorch**: 2.4.x (included in the base container)

### Setup Instructions

1. Start from the 24.05 NVIDIA PyTorch container.

2. Install packages in this order:
```
pip install --upgrade pip setuptools wheel
pip install packaging ninja
FLASH_ATTN_FORCE_BUILD=TRUE pip install --no-build-isolation --no-binary=flash-attn flash-attn==2.4.2
pip install -r requirements.txt
```

3. Do not reinstall `torch/torchvision/torchaudio` from separate CUDA wheels when using `24.05-py3`.

### Recommended module versions (24.05 baseline)

- `flash-attn==2.4.2` (source build recommended on `24.05-py3`)
- `mamba-ssm==1.2.0.post1`
- `transformers==4.41.2`
- `causal-conv1d==1.6.1`
- `opencv-python-headless==4.7.0.72`
- `triton`: use the version bundled with the container/PyTorch (do not pin separately)

### Training Instructions
---
Train with the default hyperparameters (the configuration files can be found in `config_files/train.yaml`)
```
python train.py
```
If one wants to change the hyperparmeter there are two ways:

1. Edit the configuration file `config_files/train.yaml`.
2. Run the `train.py` with parameters that corresponding to `config_files/train.yaml`. e.g.,`python train.py --Models.WorldModel.Backbone Mamba`.

### Important parameters:
Drama supports three different dynamic models: _Transformer_, _Mamba_ and _Mamba-2_. It supports two type of behaviour models: _Actor-critic_ and _PPO_.


## Code references
We've referenced several other projects during the development of this code:
- [Mamba/Mamba-2](https://github.com/state-spaces/mamba)
- [STORM](https://github.com/weipu-zhang/STORM) 
- [DreamerV3](https://github.com/danijar/dreamerv3)
