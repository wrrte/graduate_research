#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python train.py --BasicSettings.Env_name "ALE/Alien-v5"
python train.py --BasicSettings.Env_name "ALE/Amidar-v5"
python train.py --BasicSettings.Env_name "ALE/Asterix-v5"
python train.py --BasicSettings.Env_name "ALE/PrivateEye-v5"
python train.py --BasicSettings.Env_name "ALE/Seaquest-v5"