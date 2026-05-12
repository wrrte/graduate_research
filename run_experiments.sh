#!/bin/bash
export CUDA_VISIBLE_DEVICES=1



python train.py --BasicSettings.Env_name "ALE/Seaquest-v5"
python train.py --BasicSettings.Env_name "ALE/Alien-v5"
python train.py --BasicSettings.Env_name "ALE/Amidar-v5"


python train.py --BasicSettings.Env_name "ALE/PrivateEye-v5" # dreamerV3가 drama보다 점수가 높은 게임

python train.py --BasicSettings.Env_name "ALE/Asterix-v5"        # dreamerV3가 drama보다 점수가 낮은 게임

python train.py --BasicSettings.Env_name "ALE/Qbert-v5"      # dreamerV3가 drama보다 점수가 높은 게임
python train.py --BasicSettings.Env_name "ALE/Breakout-v5"   # dreamerV3가 drama보다 점수가 높음 게임
python train.py --BasicSettings.Env_name "ALE/BankHeist-v5"  # dreamerV3가 drama보다 점수가 높은 게임

python train.py --BasicSettings.Env_name "ALE/Freeway-v5"        # dreamerV3가 drama보다 점수가 낮은 게임
python train.py --BasicSettings.Env_name "ALE/ChopperCommand-v5" # dreamerV3가 drama보다 점수가 낮은 게임
python train.py --BasicSettings.Env_name "ALE/MsPacman-v5"       # dreamerV3가 drama보다 점수가 낮은 게임