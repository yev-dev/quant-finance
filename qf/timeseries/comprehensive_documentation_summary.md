# Comprehensive Documentation for Non-Linear Gap-Filling Methods

## Overview

This documentation explains the non-linear gap-filling methods implemented in `gap_filling_v2.py`, designed specifically for reconstructing distorted equity price data during financial crises. Unlike traditional linear approaches, these methods utilize non-linear models that better capture complex peer relationships during volatile periods.

## Core Non-Linear Reconstruction Methods

### 1. Static Non-Linear Reconstruction
- **Purpose**: Single-model approach using entire pre-distress window
- **Characteristics**: Fast, simple, but static adaptation
- **Best Use Cases**: Short disruptions (<15d)
- **Models Used**: RF, GBR, SVR, KNN, MLP, etc.  
- **Function**: `static_nonlinear_reconstruction(...)`

### 2. Dynamic Non-Linear Reconstruction  
- **Purpose**: Daily retraining with rolling window
- **Characteristics**: Adapts to changing market conditions
- **Best Use Cases**: Long disruptions (≥20d) with volatility spikes
- **Models Used**: All non-linear methods
- **Function**: `dynamic_nonlinear_reconstruction(...)`

### 3. ML Non-Linear Proxy Reconstruction
- **Purpose**: Feature-based approach using engineered variables
- **Characteristics**: Rich feature engineering, high performance
- **Best Use Cases**: Volatile periods with complex regime shifts  
- **Models Used**: Random Forest on engineered features
- **Function**: `ml_nonlinear_proxy_reconstruction(...)`

### 4. Ensemble Non-Linear Reconstruction
- **Purpose**: Averaged predictions from multiple models
- **Characteristics**: Robust, reduces variance
- **Best Use Cases**: When maximum accuracy is required
- **Models Used**: RF + GBR + SVR + KNN
- **Function**: `ensemble_nonlinear_reconstruction(...)`

### 5. Hyperparameter Optimization  
- **Purpose**: Random search to find best parameters
- **Characteristics**: Time intensive but optimal performance
- **Best Use Cases**: When fine-tuning is critical
- **Models Used**: All model types with optimization
- **Function**: `nonlinear_optimisation(...)`

## Visual Summary

### Algorithmic Flow Diagram
```
Data Preprocessing
    ↓
[Pre-Distress Return Window] → [Peer Selection: Spearman + DistCorr + Volatility]
    ↓  
[Model Choice: RF, GBR, KNN, SVR, etc.]
    ↓
[Reconstruction: Method-dependent Approach]
    ↓
[Evaluation: RMSE, MAE, Correlation]
    ↓
[Drift Correction: Bias, Rolling, CUSUM]
    ↓
[Backtest: Mean-Reversion Strategy P&L]
```

### Method Comparison Matrix

| Method | Advantages | Disadvantages | Best Use Case |
|--------|------------|---------------|----------------|
| **Static** | Fast, simple | Doesn't adapt to regime changes | Short disruptions (< 15d) |
| **Dynamic** | Adapts to changing markets | More computation required | Long disruption (≥ 20d) with vol spikes |
| **ML Proxy** | Rich features, high performance | Complex engineering needed | Volatile periods, complex regime shifts |
| **Ensemble** | Robust, reduces variance | Most computationally expensive | Need maximum accuracy |
| **Optimisation** | Finds best parameters | Time intensive | When fine-tuning is critical |

## Key Innovations

1. **Non-linear relationships**: Capture peer interactions that linear modeling misses
2. **Regime adaptation**: Dynamic methods adapt to changing conditions  
3. **Ensemble robustness**: Combining multiple approaches reduces variance
4. **Feature engineering**: ML approach extracts valuable financial information beyond basic correlations

## Implementation Notes

### Import Requirements
```python
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
```

### Key Model Types Available
- "rf" - Random Forest  
- "gbr" - Gradient Boosting
- "svr" - Support Vector Regression (RBF kernel)
- "knn" - K-Nearest Neighbors
- "kernel_ridge" - Kernel Ridge Regression
- "mlp" - Multi-Layer Perceptron
- "extra_trees" - Extra Trees Regressor

## Performance Comparison  

Based on empirical testing in distress scenarios:
1. **Ensemble methods** typically show the best overall accuracy
2. **Static RF** works well for short disruptions  
3. **Dynamic methods** adapt better to long periods with volatility spikes
4. **SVR** handles outliers better than most other methods
5. **ML proxy** provides superior results in complex regime shifts

## Conclusions

The non-linear gap-filling approaches provide significant improvements over traditional linear methods because they:
- Capture peer relationships that are inherently non-linear during crises
- Adapt to changing market conditions through dynamic retraining  
- Reduce variance through ensemble methods
- Better handle tail dependence and regime changes
- Provide more robust estimators for distorted price data

The choice of method should be based on the duration of disruption, available computational resources, and required accuracy levels.