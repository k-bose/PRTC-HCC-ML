# PRTC-HCC-ML
Reproducing MLs models from the Research Paper "Predicting Response to Transarterial Chemoembolization in Hepatocellualar Carcinoma Using Machine Learning Models"

# Dataset
- 1218 feature columns + 1 taget column
- 75 feature vectors
- 90-10 dataset split
    - 67 train Cases
    - 8 test Cases
- Default hyper-parameters for model training

# Results
|| RF | SVM | LR | GB | XGB |
|---|---|---|---|---|---|
|Accuracy| 0.6250 | 0.6250 | 0.3750 | 0.5000 | 0.3750 |
|F1 Score| 0.7273 | 0.7692 | 0.2857 | 0.3333 | 0.2857 |
|AUC Score| 0.6000 | 0.4667 | 0.4667 | 0.3333 | 0.4000 |
|Senstivity| 0.8000 | 1.0000 | 0.2000 | 0.2000 | 0.2000 |
|Specificity| 0.3333 | 0.0000 | 0.6667 | 1.0000 | 0.6667 |