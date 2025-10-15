# Jet Engine Fault Detection

A machine learning project to detect jet engine faults from sensor data. It includes EDA, model training (Decision Tree, RandomForest, XGBoost/LightGBM), persisted pipelines, and a Streamlit app for predictions.

## Project Structure
- `engine_fault_detection_dataset.csv` – source dataset
- `Untitled.ipynb` – full EDA + modeling + CV + save best model
- `DecisionTree_EngineFault.ipynb` – EDA + tuned Decision Tree
- `DecisionTree_Basic.ipynb` – basic EDA + simple Decision Tree
- `artifacts/` – saved models (`jet_fault_model_*.pkl`) and `feature_metadata.pkl`
- `app.py` – Streamlit frontend (upload CSV → predictions)
- `requirements.txt` – dependencies
- `sample_engine_fault_data.csv` – tiny labeled sample
- `generate_unlabeled_csv.py` – makes unlabeled samples
- `sample_engine_fault_unlabeled.csv` – 300-row unlabeled sample

## Setup
1) Create environment
```bash
conda create -n jetfault python=3.12 -y
conda activate jetfault
```
2) Install deps
```bash
pip install -r requirements.txt
```

## Train Models
Pick a notebook and run top-to-bottom:
- `Untitled.ipynb` (RF/XGB/LGBM, CV, select best → saves to `artifacts/`)
- `DecisionTree_EngineFault.ipynb` (tuned DT)
- `DecisionTree_Basic.ipynb` (fast DT)

Artifacts saved:
- `artifacts/jet_fault_model_<name>.pkl`
- `artifacts/feature_metadata.pkl` with `numeric_features`, `target`, `classes_`

## Run Frontend
```bash
python -m streamlit run app.py
```
- Upload CSV with feature columns only (no `Engine_Condition`).
- App shows preview, predictions, and probabilities.
- If multiple models exist, the app loads the first `jet_fault_model_*.pkl`; remove others if needed.

## Test Files
- Labeled sample: `sample_engine_fault_data.csv`
- Unlabeled sample: `sample_engine_fault_unlabeled.csv`
- Generate a new unlabeled set:
```bash
python generate_unlabeled_csv.py  # default 300 rows
```

## What Was Done (Steps)
- EDA: shape/info/nulls/duplicates; class balance; histograms; correlations; boxplots
- Preprocessing: numeric scaling with `StandardScaler` in a `ColumnTransformer`
- Modeling: DT baseline; RF/XGBoost/LightGBM with tuning (macro-F1 focus)
- Validation: holdout metrics + 5-fold CV; model selection by macro F1
- Persistence: best pipeline + metadata saved to `artifacts/`
- Frontend: Streamlit uploader → applies pipeline → outputs class and `predict_proba`

Notes:
- DT probabilities can be hard (0/1). For softer probs: set `max_depth`, increase `min_samples_leaf/min_samples_split`.
- For class imbalance, use SMOTE/class weights (see `Untitled.ipynb`).

## Troubleshooting
- Streamlit not found → `pip install -r requirements.txt`, run with `python -m streamlit run app.py`.
- Schema errors → ensure your CSV has exactly the columns in `feature_metadata.pkl` (`numeric_features`), order will be enforced in-app.

License: educational use only.
