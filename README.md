# How Does Pruning Impact Long-Tailed Multi-Label Medical Image Classifiers?

[**Gregory Holste**](https://gholste.me), Ziyu Jiang, Ajay Jaiswal, Maria Hanna, Shlomo Minkowitz, Alan C. Legasto, Joanna G. Escalon, Sharon Steinberger, Mark Bittman, Thomas C. Shen, Ying Ding, Ronald M. Summers, George Shih, Yifan Peng, and Zhangyang Wang

[**[arXiv]**](https://arxiv.org/abs/2308.09180) | Early accepted (*top 14%*) to [MICCAI 2023](https://conferences.miccai.org/2023/en/)

### NOTE: This repository is a work in progress! ###

-----

# Abstract

<p align=center>
    <img src=figs/class_prune_correlations.png height=500>
</p>

Pruning has emerged as a powerful technique for compressing deep neural networks, reducing memory usage and inference time without significantly affecting overall performance. However, the nuanced ways in which pruning impacts model behavior are not well understood, particularly for long-tailed, multi-label datasets commonly found in clinical settings. This knowledge gap could have dangerous implications when deploying a pruned model for diagnosis, where unexpected model behavior could impact patient well-being. To fill this gap, we perform the first analysis of pruning’s effect on neural networks trained to diagnose thorax diseases from chest X-rays (CXRs). On two large CXR datasets, we examine which diseases are most affected by pruning and characterize class “forgettability” based on disease frequency and co-occurrence behavior. Further, we identify individual CXRs where uncompressed and heavily pruned models disagree, known as pruning-identified exemplars (PIEs), and conduct a human reader study to evaluate their unifying qualities. We find that radiologists perceive PIEs as having more label noise, lower image quality, and higher diagnosis difficulty. This work represents a first step toward understanding the impact of pruning on model behavior in deep long-tailed, multi-label medical image classification. All code, model weights, and data access instructions can be found at https://github.com/VITA-Group/PruneCXR.

# Dataset Access

## COMING SOON ##

# Trained Models

All trained models can be found at https://utexas.box.com/s/eenr2gumfjaazngvnwerhmo9c0qru23w under the `trained_models` directory. In this directory, you will find 60 subdirectories, representing the 30 "runs" of a ResNet50 trained on each dataset (NIH-CXR-LT and MIMIC-CXR-LT) as described in the paper. Each subdirectory contains the weights, training history, and evaluation metrics for a given run.

All test set predictions for each pruned model (used to generate all main figures and results in the paper) can be found at https://utexas.box.com/s/eenr2gumfjaazngvnwerhmo9c0qru23w under the `nih-cxr-lt_L1-prune_preds` and `mimic-cxr-lt_L1-prune_preds` directories. Each directory contains a `.pkl` file containing a NumPy array of test set predictions for each combination of *seed* $s \in \{0, 1, \dots, 29\}$ and *sparsity ratio* $k \in \{0, 0.05, \dots, 0.9, 0.95\}$ for the given dataset (NIH-CXR-LT or MIMIC-CXR-LT).

# Usage

To reproduce the results in the paper:

1. Register to download the MIMIC-CXR-JPG dataset from https://physionet.org/content/mimic-cxr-jpg/2.0.0/, and download the NIH ChestXRay14 dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC/.
2. Download NIH-CXR-LT labels from **COMING SOON** and MIMIC-CXR-LT labels from **COMING SOON**. Place all 6 files in the `labels` directory of this repo.
3. Install prerequisite packages with Anaconda: `conda env create -f torch.yml` and `conda activate torch`.
4. (Optional) From https://utexas.box.com/s/eenr2gumfjaazngvnwerhmo9c0qru23w, download the `trained_models` directory and place it in this repo's base directory. The script to train these models can be found in `src/run_experiments.sh`. 
5. From https://utexas.box.com/s/eenr2gumfjaazngvnwerhmo9c0qru23w, download `nih-cxr-lt_L1-prune_preds` and `mimic-cxr-lt_L1-prune_preds` and place both directories in this repo's base directory. The script to perform L1 pruning and inference on all models trained in the previous step can be found in `src/prune.py`.
6. Generate the content of Figures 1-4 by running `python src/main_analysis.py`.
7. Generate the content of Figure 5 by running `python src/pie_analysis.py`.

While the raw data from the radiologist reader study described in Section 2.3 is not publicly available, the code to generate Figure 6, which summarizes radiologist perception of PIEs, can be found in `src/pie_survey_analysis.py`. 

# Contact

Please either open a Github Issue or reach out to Greg Holste at [gholste@utexas.edu](mailto:gholste@utexas.edu) with any questions.
