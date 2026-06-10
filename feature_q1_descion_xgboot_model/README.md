# SmartBuild Weight Prediction

A machine learning project to predict product weight using only pre-production data. Compares Decision Tree vs XGBoost.

## What this does

Predicts the weight (in kg) of manufactured products before production starts. The tricky part was making sure we only use features that are actually available before production - otherwise we'd be cheating (data leakage).

## Features used

We use 9 features that are known before production:
- `width`, `height` - input dimensions
- `pressure`, `karma`, `modulation` - machine settings
- `ionizationclass`, `FluxCompensation` - more settings
- `gear`, `rotation_speed` - machine config

We explicitly exclude post-production stuff like `distortion`, `roughness`, `Quality`, etc. Found that `nicesness` had 99.5% correlation with width - definitely measured after production.

## Dataset

- `Production_Log_01.csv` - 10,000 production records
- `Machine_Settings_Log_01.csv` - machine config data (joins on `configuration_log_ID`)

## How to run

```bash
# install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib

# run it
python smartbuild_weight_predictor.py
```

## Results

| Model | R2 Score | MAE |
|-------|----------|-----|
| Baseline (just mean) | -0.00 | 150,584 kg |
| Decision Tree | 0.9964 | 7,349 kg |
| XGBoost | 0.9992 | 2,695 kg |

XGBoost wins with 99.92% variance explained.

Top features by importance:
1. width (~97%)
2. height (~1-2%)
3. ionizationclass (~0.5-0.8%)

Both models pass the overfitting check - train/test gap is under 1%.

## Output files

The script generates 4 plots:
- `model_comparison_visualization.png` - 4-panel comparison
- `residuals_analysis.png` - prediction errors
- `cross_validation_scores.png` - CV score distributions
- `metrics_heatmap.png` - performance summary

## Model parameters

Decision Tree:
- max_depth=10
- min_samples_split=20
- min_samples_leaf=10

XGBoost:
- n_estimators=100
- max_depth=6
- learning_rate=0.1

## Data cleaning

- Outliers removed using IQR method (Q1-3*IQR to Q3+3*IQR)
- Categorical encoding: ionizationclass (A/B/C -> 0/1/2), FluxCompensation (I-IV -> 1-4)
- 80/20 train/test split
- 5-fold cross-validation

## Notes

- Width is by far the most important predictor - makes sense since bigger products weigh more
- The high R2 scores are legit - no data leakage since we only use pre-production features
- XGBoost takes a bit longer to train but gives better results
