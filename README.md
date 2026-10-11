# Data-Driven Maintenance Prioritisation using Machine Performance and Health Data

This repository contains the executable implementation used for the thesis project on explainable and capacity-aware predictive maintenance using the **AI4I 2020 Predictive Maintenance Dataset**.

The implementation is centred on the notebook:


The workflow covers dataset auditing, leakage-safe feature preparation, baseline modelling, imbalance-aware model screening, stratified cross-validation, hyperparameter optimisation, probability-threshold selection, calibration assessment, bootstrap uncertainty estimation, SHAP explainability, maintenance-priority construction, and an interactive Plotly dashboard.

## 1. Research objective

The project develops a reproducible machine-learning decision-support workflow that:

1. predicts the binary **Machine failure** outcome;
2. avoids target leakage from failure-mode indicators and identifiers;
3. compares baseline and tree-based classifiers under severe class imbalance;
4. selects operating thresholds from development-only out-of-fold predictions;
5. evaluates probability calibration;
6. explains the selected Random Forest using SHAP;
7. converts failure probabilities into a ranked maintenance-review queue; and
8. generates machine-readable outputs and an interactive dashboard.

The system is a **decision-support prototype**, not an autonomous maintenance controller.

## 2. Dataset

### Canonical dataset

**AI4I 2020 Predictive Maintenance Dataset**  
UCI Machine Learning Repository  
DOI: `10.24432/C5HS5C`

### Dataset retrieval used in the notebook

The first notebook cell uses KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
)
```

The implemented dataset contains:

- **10,000 observations**
- **14 original columns**
- **9,661 non-failure observations**
- **339 failure observations**
- failure prevalence of **3.39%**
- no missing values in the modelling variables
- no duplicate rows reported by the notebook

The data are synthetic benchmark data. They do not contain human participants or personal information.

## 3. Target, predictors, and leakage control

### Prediction target

`Machine failure`

### Predictors used by the model

The modelling matrix contains exactly eight pre-outcome predictors:

1. `Type`
2. `Air temperature [K]`
3. `Process temperature [K]`
4. `Rotational speed [rpm]`
5. `Torque [Nm]`
6. `Tool wear [min]`
7. `Temperature difference [K]`
8. `Power [W]`

### Engineered features

`Temperature difference [K]` is calculated as:

```text
Process temperature [K] - Air temperature [K]
```

`Power [W]` is calculated from torque and angular velocity:

```text
Power = Torque × (Rotational speed × 2 × π / 60)
```

### Variables deliberately excluded from predictive modelling

The following variables are excluded from the feature matrix:

- `UDI`
- `Product ID`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

`UDI` and `Product ID` are identifiers. The five failure-mode indicators are retained only for descriptive auditing because they contain outcome-related failure information and would introduce target leakage if used as predictors.

## 4. Preprocessing

The notebook uses a `ColumnTransformer` inside scikit-learn pipelines.

### Numerical variables

The seven numerical predictors are processed with:

```python
StandardScaler()
```

### Categorical variable

`Type` is processed with:

```python
OneHotEncoder(
    handle_unknown="ignore",
    drop="first"
)
```

All learned preprocessing is performed inside the modelling pipeline so transformations are learned from training data rather than from the full dataset before validation.

## 5. Data split and validation design

The notebook uses a stratified 80:20 development/holdout split:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

This produces:

- development set: **8,000 observations**, including **271 failures**
- holdout set: **2,000 observations**, including **68 failures**

Five-fold stratified cross-validation is used with:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The primary optimisation metric is **PR-AUC**, implemented through scikit-learn's `average_precision` score. Precision, recall, F1-score, ROC-AUC, specificity, confusion-matrix counts, calibration scores, and false-positive behaviour are also reported.

### Important validation limitation

The 2,000-case holdout partition is used during exploratory algorithm screening. It is therefore **not a completely untouched one-time final test set**. Formal tuning, calibration comparison, and operational-threshold selection are development-only, but later holdout results should still be interpreted as conditional internal evaluation rather than a pristine independent test.

## 6. Models implemented

### Baselines

- Dummy Classifier
- Logistic Regression

### Exploratory imbalance-aware screening

- Balanced Logistic Regression
- Decision Tree
- Balanced Decision Tree
- Random Forest
- Balanced Random Forest
- Gradient Boosting
- XGBoost
- Balanced XGBoost

### Models formally cross-validated and tuned

- Random Forest
- Gradient Boosting
- XGBoost

Hyperparameter optimisation uses `RandomizedSearchCV` with:

- 5-fold stratified cross-validation
- `n_iter=30` per model
- `scoring="average_precision"`
- `random_state=42`
- `n_jobs=-1`

## 7. Hyperparameter search spaces

### Random Forest

```text
n_estimators:        200, 300, 500, 700
max_depth:           None, 6, 10, 15, 20
min_samples_split:   2, 5, 10
min_samples_leaf:    1, 2, 4
max_features:        sqrt, log2, None
class_weight:        None, balanced, balanced_subsample
```

Best notebook configuration:

```text
n_estimators = 700
max_depth = 10
min_samples_split = 2
min_samples_leaf = 2
max_features = None
class_weight = None
```

Best cross-validated PR-AUC reported by the search: **0.8859**

### Gradient Boosting

```text
n_estimators:        50, 100, 150, 200, 300
learning_rate:       0.01, 0.03, 0.05, 0.10, 0.20
max_depth:           1, 2, 3, 4, 5
min_samples_split:   2, 5, 10
min_samples_leaf:    1, 2, 4
subsample:           0.7, 0.8, 0.9, 1.0
max_features:        None, sqrt, log2
```

Best notebook configuration:

```text
subsample = 0.9
n_estimators = 150
min_samples_split = 10
min_samples_leaf = 4
max_features = log2
max_depth = 5
learning_rate = 0.03
```

Best cross-validated PR-AUC reported by the search: **0.8904**

### XGBoost

```text
n_estimators:        100, 200, 300, 500
learning_rate:       0.01, 0.03, 0.05, 0.10
max_depth:           2, 3, 4, 5, 6
min_child_weight:    1, 3, 5, 7
subsample:           0.7, 0.8, 0.9, 1.0
colsample_bytree:    0.7, 0.8, 0.9, 1.0
gamma:               0, 0.1, 0.3, 0.5
reg_lambda:          1, 2, 5, 10
reg_alpha:           0, 0.01, 0.1, 0.5
```

Best notebook configuration:

```text
subsample = 0.8
reg_lambda = 5
reg_alpha = 0.5
n_estimators = 200
min_child_weight = 1
max_depth = 6
learning_rate = 0.05
gamma = 0.3
colsample_bytree = 0.8
```

Best cross-validated PR-AUC reported by the search: **0.8610**

## 8. Final downstream model

The notebook carries the **tuned Random Forest** forward into threshold selection, calibration assessment, uncertainty analysis, SHAP explanations, maintenance ranking, and dashboard generation.

At the default 0.50 threshold, the tuned Random Forest reports:

| Metric | Value |
|---|---:|
| Accuracy | 0.9920 |
| Precision | 0.9483 |
| Recall | 0.8088 |
| F1-score | 0.8730 |
| ROC-AUC | 0.9838 |
| PR-AUC | 0.8985 |
| False positives | 3 |
| False negatives | 13 |
| True positives | 55 |
| True negatives | 1,929 |

## 9. Development-derived operating thresholds

Threshold selection is repeated using development-only out-of-fold probabilities.

The implemented thresholds are:

- **High-priority threshold:** `0.42`
- **Medium-priority lower threshold:** `0.05`
- probabilities below `0.05` are labelled Low priority

The out-of-fold 0.42 threshold is selected as the best-F1 operating threshold. On the reused 2,000-case holdout it gives:

| Metric | Value |
|---|---:|
| Accuracy | 0.99200 |
| Precision | 0.93333 |
| Recall | 0.82353 |
| Specificity | 0.99793 |
| F1-score | 0.87500 |
| ROC-AUC | 0.98378 |
| PR-AUC | 0.89847 |
| Brier score | 0.00793 |
| True negatives | 1,928 |
| False positives | 4 |
| False negatives | 12 |
| True positives | 56 |

Bootstrap uncertainty is estimated with **2,000 resamples** using NumPy's random generator with seed `42`.

## 10. Probability calibration

The tuned Random Forest is compared using development-only out-of-fold probabilities against:

- raw Random Forest probabilities
- sigmoid calibration
- isotonic calibration

Development OOF Brier scores reported by the notebook are:

| Method | Brier score |
|---|---:|
| Raw Random Forest | 0.00845 |
| Sigmoid | 0.00875 |
| Isotonic | 0.00853 |

The raw Random Forest probability output is retained because it has the lowest development OOF Brier score among the three compared probability mappings.

## 11. SHAP explainability

The final Random Forest is explained using `shap.TreeExplainer`.

The transformed holdout matrix contains nine columns after one-hot encoding:

- seven numerical predictors
- `Type_L`
- `Type_M`

The notebook reports the following global mean absolute SHAP importance:

| Rank | Feature | Relative importance |
|---:|---|---:|
| 1 | Rotational speed [rpm] | 25.21% |
| 2 | Tool wear [min] | 23.37% |
| 3 | Temperature difference [K] | 21.00% |
| 4 | Power [W] | 20.09% |
| 5 | Torque [Nm] | 6.03% |
| 6 | Type_L | 1.56% |
| 7 | Process temperature [K] | 1.43% |
| 8 | Air temperature [K] | 0.98% |
| 9 | Type_M | 0.33% |

SHAP values are interpreted as **model attributions**, not causal engineering diagnoses.

## 12. Implemented maintenance-priority layer

The current notebook constructs threshold-based maintenance groups:

| Priority | Probability rule | Machines | Actual failures |
|---|---|---:|---:|
| High | `p >= 0.42` | 60 | 56 |
| Medium | `0.05 <= p < 0.42` | 93 | 8 |
| Low | `p < 0.05` | 1,847 | 4 |

The maintenance table includes:

- maintenance rank
- UDI
- Product ID
- product type
- failure probability
- 0–100 maintenance risk score
- priority
- recommended action
- primary SHAP risk driver
- driver direction
- operating variables

Recommended actions are generated as:

```text
High   -> Immediate inspection / maintenance review
Medium -> Increase monitoring and schedule inspection
Low    -> Continue routine monitoring
```

## 13. Current implementation boundary: fixed inspection capacities

The thesis discusses exact fixed inspection capacities of:

- 5% = 100 of 2,000 holdout cases
- 10% = 200 cases
- 20% = 400 cases

However, the supplied `UC_17717_updated.ipynb` **does not currently execute and save the exact top-k 5%, 10%, and 20% capacity metric table**. Its implemented maintenance decision layer is threshold-based using `0.42` and `0.05`.

Accordingly:

- `maintenance_priority_validation.csv` should be described as a validation file for the implemented ranked/threshold priority output;
- it should not be described as the saved 5%, 10%, and 20% fixed-capacity result table;
- exact top-k capacity metrics should only be claimed as reproducible repository outputs after a dedicated top-k calculation is added and rerun.

This distinction is intentionally recorded to keep the repository consistent with the supplied implementation.

## 14. Generated output files

Running the notebook produces the following repository-ready outputs.

### `maintenance_dashboard_dataset.csv`

Contains the decision-support fields used by the dashboard:

```text
Maintenance Rank
UDI
Product ID
Type
Failure Probability
Maintenance Risk Score
Maintenance Priority
Recommended Action
Primary Risk Driver
Driver Effect
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
Temperature difference [K]
Power [W]
```

The observed target is intentionally excluded from the operational dashboard table.

### `maintenance_priority_validation.csv`

Contains:

```text
Maintenance Rank
UDI
Product ID
Failure Probability
Maintenance Priority
Machine failure
```

This file retains the observed target for retrospective validation.

### `global_shap_feature_importance.csv`

Contains the final global SHAP feature-importance table.

### `predictive_maintenance_dashboard.html`

Self-contained interactive Plotly dashboard generated from the model-derived maintenance dataset.

The dashboard includes:

- maintenance-priority distribution
- maintenance-risk distribution
- top-ranked maintenance queue
- primary risk-driver summaries
- product-type comparison
- operating-condition map
- probability-versus-rank curve
- interactive maintenance queue table

## 15. Software environment

The notebook metadata identifies **Python 3** and a Google Colab **T4 GPU** session, but it does not store a historical package-version manifest.

Therefore, the included `requirements.txt` and `environment.yml` provide a **pinned reference reproducibility environment** for this repository. They are not presented as a claim that every package version exactly matches the original historical Colab image.

A GPU is not required by the notebook's implemented modelling code.

## 16. Installation

### Option A: Conda environment

```bash
conda env create -f environment.yml
conda activate explainable-machine-failure
jupyter lab
```

Open:

```text
UC_17717_updated.ipynb
```

and run cells from top to bottom.

### Option B: Python virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

## 17. Dataset-path requirement

The notebook currently contains two dataset-path mechanisms:

1. the first cell downloads the dataset with `kagglehub.dataset_download(...)`;
2. Step 1 later assigns:

```python
path = "/kaggle/input/predictive-maintenance-dataset-ai4i-2020"
```

### Running on Kaggle

Attach the Kaggle dataset named by the notebook. The hard-coded `/kaggle/input/...` path can then be used as written.

### Running locally or in Google Colab

After the KaggleHub download, replace the hard-coded Step 1 path with the directory returned by KaggleHub.

For example:

```python
dataset_dir = kagglehub.dataset_download(
    "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
)

path = dataset_dir
```

The Step 1 loader automatically searches that directory for `.csv` files and loads the first CSV file found.

## 18. Reproducibility controls

The notebook fixes the main random processes with seed `42`, including:

- train/holdout split
- stratified cross-validation
- RandomizedSearchCV
- applicable scikit-learn estimators
- XGBoost models
- bootstrap random-number generation

The following settings should remain unchanged when attempting to reproduce the reported notebook outputs:

```text
train/holdout split: 80/20 stratified
random_state: 42
cross-validation: 5-fold StratifiedKFold
CV shuffle: True
RandomizedSearchCV iterations: 30 per tuned model
primary tuning score: average_precision
bootstrap repetitions: 2000
OOF high-priority threshold: 0.42
OOF medium-priority lower threshold: 0.05
```

Small numerical differences can still occur across operating systems, processor architectures, BLAS/OpenMP implementations, or future dependency changes. The pinned environment files are included to minimise these differences.

## 19. Suggested repository structure

```text
Explainable-Machine-Failure-Prediction/
|
|-- UC_17717_updated.ipynb
|-- README.md
|-- requirements.txt
|-- environment.yml
|
|-- maintenance_dashboard_dataset.csv          # generated after execution
|-- maintenance_priority_validation.csv        # generated after execution
|-- global_shap_feature_importance.csv          # generated after execution
|-- predictive_maintenance_dashboard.html      # generated after execution
```

## 20. Reproduction checklist

A repository reproduction is complete when all of the following are confirmed:

- the AI4I dataset loads as 10,000 rows and 14 original columns;
- `Machine failure` contains 339 positive observations;
- excluded identifiers and failure-mode indicators do not enter the predictive matrix;
- the modelling matrix contains the eight documented predictors;
- the split contains 8,000 development and 2,000 holdout observations;
- cross-validation and model tuning execute using seed 42;
- the tuned Random Forest is available as `best_rf_model`;
- the OOF operating threshold is 0.42;
- raw Random Forest probabilities remain the selected probability mapping;
- bootstrap uncertainty uses 2,000 resamples;
- SHAP is calculated only from the admissible transformed predictors;
- the three CSV files are generated;
- `predictive_maintenance_dashboard.html` is generated;
- any future fixed 5%, 10%, and 20% top-k capacity results are saved separately and are not confused with threshold-based priority groups.

## 21. Scope and limitations

This repository is an analytical benchmark and decision-support prototype. Important limitations are:

- the AI4I dataset is synthetic;
- the holdout partition was viewed during exploratory screening;
- the reported holdout is therefore not a pristine one-time external validation set;
- SHAP explanations show model attribution rather than causality;
- the dashboard does not perform live sensor ingestion;
- the dashboard does not automatically schedule maintenance;
- asset criticality, workforce scheduling, spare-parts allocation, intervention timing, and failure-consequence costs are outside the implemented model;
- exact fixed-capacity 5%, 10%, and 20% top-k outputs are not currently saved by the supplied notebook.

## 22. Code repository

Repository referenced in the thesis:

`https://github.com/Subhamresearch2026/Explainable-Machine-Failure-Prediction`
