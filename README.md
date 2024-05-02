# Measuring Strength of Joint Causal Effects
In this repo, we provide MATLAB code to reproduce the results from our paper "[Measuring Strength of Joint Causal Effects](https://doi.org/10.1109/TSP.2024.3394660)," published in the IEEE Transactions on Signal Processing.

> **Abstract:** In the study of causality, we often seek not only to detect the presence of cause-effect relationships, but also to characterize how multiple causes combine to produce an effect. When the response to a change in one of the causes depends on the state of another cause, we say that there is an interaction or joint causation between the multiple causes. In this paper, we formalize a theory of joint causation based on higher-order derivatives and causal strength. Our proposed measure of joint causal strength is called the mixed differential causal effect (MDCE). We show that the MDCE approach can be naturally integrated into existing causal inference frameworks based on directed acyclic graphs or potential outcomes. We then derive a non-parametric estimator of the MDCE using Gaussian processes. We validate our approach with several experiments using synthetic data sets, demonstrating its applicability to static data as well as time series. 


## Instructions
To generate all figures (as .png files), you just need to run `main.m`. The code should run with no issues using Matlab 2022a or later. All generated figures and tables will be saved to the results folder. 
```
git clone https://github.com/KurtButler/joint_causation
```
If you wish to reproduce our example that uses the [New Taipei City housing data](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set), you will additionally need to download the data set from the UCI Machine Learning Repository and put it in the `data` folder. 

## Data Availability
In our experiments, we used a publicly available data set from the UCI Machine Learning Repository:
- [New Taipei City Housing Data](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set), donated by I-Cheng Yeh at the Department of Civil Engineering, Tamkang University

## Citation
If you use any code or results from this project in your academic work, please cite our paper:
```
@article{butler2024joint,
  title={Measuring Strength of Joint Causal Effects},
  author={Butler, Kurt and Feng, Guanchao and Djuri{\'c}, Petar M},
  journal={IEEE Transactions on Signal Processing},
  year={2024},
  publisher={IEEE},
  doi={10.1109/TSP.2024.3394660}
}
```
