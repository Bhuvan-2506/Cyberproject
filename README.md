# Intrusion Detection System (IDS) using Machine Learning

## Overview
This is a beginner-friendly IDS project built using **RandomForest** on the **NSL-KDD dataset**.  
The system classifies network traffic as **normal** or **attack**.

## Dataset
- [NSL-KDD Dataset](https://github.com/jmnwong/NSL-KDD-Dataset)  
- Files used: `KDDTrain+.TXT` and `KDDTest+.TXT`  

## Features
- Binary classification: normal vs attack
- RandomForest model with 150 trees
- Feature scaling and SMOTE for class balancing
- Generates:
  - Confusion matrix (`confusion_matrix.png`)
  - Top feature importances (`feature_importances.png`)
  - Evaluation report (`rf_eval_report.txt`)

## How to Run
1. Clone or download this repository.
2. Download the NSL-KDD dataset and place `KDDTrain+.TXT` and `KDDTest+.TXT` in the project folder.
3. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows CMD
   # or for PowerShell: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   # then: venv\Scripts\Activate.ps1
