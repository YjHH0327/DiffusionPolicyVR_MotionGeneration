# DiffusionPolicyVR_MotionGeneration


## Overview
This repository contains the implementation of our research on applying Diffusion Policy to motion sequence generation in Virtual Reality. This project is based on this [paper](https://diffusion-policy.cs.columbia.edu/)

## Visualization of synthetic data
https://private-user-images.githubusercontent.com/122988291/415734704-8b438cb7-ac64-4c26-8ffc-b2443aeb7970.mp4

## Dataset
You can download data files [here](https://drive.google.com/file/d/10UwetbvUrSrcXK6YW29MIR3z8Hewz3oZ/view?usp=sharing). Then put the unzipped files into ./data

We provide our scene configuration file in:
```
./data/data_diffusion_vr/laby_config/laby_config.json
```
The motion dataset will be released soon. For test, you can put your own .zarr dataset here:
```
./data/data_diffusion_vr/my_zaar_shortstart_absolu_relative_origin/
```

## Conda environment
```
$ conda env create -f conda_environment.yaml
$ conda activate robodiff
```

## Train a Diffusion Policy
```
$ DISPLAY=:0 HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=train_diffusion_transformer_lowdim_vr_lumi_active_runner_workspace.yaml training.seed=42 training.devi
ce=cuda:0 hydra.run.dir='/workspace/rack/data/trainings_diffusion_policy/data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

## Inference with checkpoints and proposed first frame
We provide here our used checkpoints and one initial frame.
```
$ DISPLAY=:0 HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=infer_diffusion_transformer_lowdim_vr_lumi_active_runner_workspace.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='./data/trainings_diffusion_policy/data/outputs/synthetic_^Cta_from_2024.11.15/15.05.17/'
```
