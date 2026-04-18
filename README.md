# University Sports Venue Resource Scheduling based on CE-SLO Algorithm

This repository contains the official data, simulation environments, and algorithmic implementations for the paper: 
**"An Optimized Scheduling Method for University Sports Venue Resources Based on the Cultural-Exchange-based Social Learning Optimization Algorithm"**.

This project aims to solve complex discrete resource allocation problems in university sports facilities by introducing a novel Cultural-Exchange-based Social Learning Optimization (CE-SLO) algorithm.

## 📊 Dataset Availability

Because this is a simulation-based experiment, the dataset is programmatically generated. To facilitate easy access and reproducibility, the fully generated multi-week scheduling datasets have been hosted on Kaggle.

👉 **[Download the Dataset on Kaggle](https://www.kaggle.com/datasets/kzheng970/university-sports-venue-scheduling-optimization/data)**

The datasets cover 1-week to 5-week scheduling horizons, allowing for scalability testing.

## 📂 Repository Structure

The project is systematically organized into 5 main parts corresponding to the experimental phases of the paper:

### Part 1: Data Generation (`Init_Data/`)
Scripts for generating the simulated scheduling requests.
- `Init_singleweek.py`: Generates baseline single-week data.
- `Init_multiweek.py`: Generates multi-week periodic data.
- *Outputs:* `DataSet_weeks1_123.csv` to `DataSet_weeks5_123.csv`

### Part 2: Feasibility Verification (`Feasibility/`)
- `Feasibility_MultiWeek.py`: Core code for verifying the feasibility of the CE-SLO algorithm in addressing multi-week scheduling constraints.

### Part 3: Ablation Experiment (`Ablation Experiment/`)
- `Ablation_Experiment.py`: Code for conducting ablation studies to validate the effectiveness of the proposed "Cultural-Exchange" mechanism.

### Part 4: Performance Comparison (`Performance Comparison/`)
- `Performance Comparison.py`: Benchmarking the CE-SLO algorithm against other heuristic optimization algorithms.

### Part 5: Statistical Significance Analysis (`Wilcoxon-Friedman/`)
- `Wilcoxon&Friedman.py`: Statistical validation of the algorithm's performance superiority using Wilcoxon signed-rank and Friedman tests.

## ⚙️ Environment Setup

We recommend using Anaconda to manage your Python environment. You can set up the environment using the following commands:

```bash
# Create a new conda environment
conda create -n venue_scheduling python=3.12 -y

# Activate the environment
conda activate venue_scheduling
