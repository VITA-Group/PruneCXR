#!/usr/bin/bash

# Train MIMIC-CXR-LT models
for seed in {0..29}
do
    python main.py --data_dir /ssd1/greg/physionet.org/files/mimic-cxr-jpg/2.0.0 --dataset mimic-cxr-lt --out_dir ../trained_models --seed $seed
done

# Train NIH-CXR-LT models
for seed in {0..29}
do
    python main.py --data_dir /ssd1/greg/NIH_CXR/images --dataset nih-cxr-lt --out_dir ../trained_models --seed $seed
done
