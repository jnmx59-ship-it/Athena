# ATHENA

**A**daptive **T**emporal **H**armonization and **E**stimation for **N**ormalized **A**nomalies

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade machine learning framework for time series anomaly detection using cross-series regression and iterative refinement.

## Overview

ATHENA is designed to detect anomalies in multivariate time series data by learning the relationships between different time series. Unlike univariate methods that analyze each series in isolation, ATHENA leverages cross-series dependencies to identify points where observed values deviate from what would be expected given the behavior of related series.

### Key Features

- **Cross-Series Learning**: Learns predictive relationships between multiple time series
- **Temporal Harmonization**: Automatically aligns time series with different frequencies and timestamps
- **Iterative Refinement**: Uses an iterative reconstruction algorithm for robust anomaly detection
- **Flexible Models**: Works with any scikit-learn compatible regressor
- **Parallel Processing**: Built-in parallelization for efficient training and inference
- **Production Ready**: Comprehensive error handling, type hints, and documentation

## Installation

### From PyPI (when published)

```bash
pip install athena-anomaly
```

### From Source

```bash
git clone https://github.com/username/athena.git
cd athena
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/username/athena.git
cd athena
pip install -e ".[dev]"
```

## Quick Start

```python
from athena import ATHENA
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# Create sample time series data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=1000, freq='h')

data = {
    'temperature': pd.Series(
        np.sin(np.linspace(0, 20*np.pi, 1000)) * 10 + 20 + np.random.normal(0, 0.5, 1000),
        index=dates
    ),
    'humidity': pd.Series(
        np.cos(np.linspace(0, 20*np.pi, 1000)) * 20 + 60 + np.random.normal(0, 1, 1000),
        index=dates
    ),
    'pressure': pd.Series(
        np.sin(np.linspace(0, 10*np.pi, 1000)) * 5 + 1013 + np.random.normal(0, 0.3, 1000),
        index=dates
    ),
}

# Inject some anomalies
data['temperature'].iloc[100:105] += 15  # Temperature spike
data['humidity'].iloc[500:510] -= 30     # Humidity drop

# Initialize ATHENA with Random Forest
athena = ATHENA(
    model_class=RandomForestRegressor,
    model_params={
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    n_jobs=-1,
    verbose=0
)

# Fit the model
athena.fit(data, ts='1h', interpolation_method='linear')

# Detect anomalies
anomalies = athena.predict(data, n_iterations=5)

# View results
print(f"Total anomalies detected: {anomalies.sum().sum()}")
print(f"\nAnomalies by series:")
print(anomalies.sum())
```

## How It Works

### Algorithm Overview

1. **Data Harmonization**: ATHENA first aligns all input time series to a common time grid by:
   - Performing an outer join on timestamps
   - Resampling to a target frequency
   - Interpolating missing values

2. **Standardization**: Each series is z-score normalized to ensure comparable scales.

3. **Cross-Series Model Training**: For each series $X_i$, ATHENA trains a model $f_i$ that predicts $X_i$ from all other series $X_{\setminus i}$:
   
   $$\hat{X}_i = f_i(X_1, X_2, ..., X_{i-1}, X_{i+1}, ..., X_n)$$

4. **Iterative Refinement**: During prediction, ATHENA applies multiple refinement iterations where the output of iteration $k$ becomes the input for iteration $k+1$. This process converges toward a harmonious representation of the data.

5. **Anomaly Detection**: Points where the absolute residual between observed and reconstructed values exceeds 1 standard deviation are flagged as anomalies:
   
   $$\text{anomaly}_i = |X_i - \hat{X}_i| \geq 1.0$$

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ATHENA Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Series A   │    │   Series B   │    │   Series C   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│                             ▼                                   │
│                   ┌──────────────────┐                          │
│                   │   Harmonization  │                          │
│                   │   (Align & Resample)                        │
│                   └────────┬─────────┘                          │
│                            │                                    │
│                            ▼                                    │
│                   ┌──────────────────┐                          │
│                   │  Standardization │                          │
│                   └────────┬─────────┘                          │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                 │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐            │
│  │  Model_A   │    │  Model_B   │    │  Model_C   │            │
│  │ f(B,C)→A  │    │ f(A,C)→B  │    │ f(A,B)→C  │            │
│  └─────┬──────┘    └─────┬──────┘    └─────┬──────┘            │
│        │                 │                 │                    │
│        └─────────────────┼─────────────────┘                    │
│                          │                                      │
│                          ▼                                      │
│                ┌──────────────────┐                             │
│                │    Iterative     │                             │
│                │   Refinement     │                             │
│                │   (M iterations) │                             │
│                └────────┬─────────┘                             │
│                         │                                       │
│                         ▼                                       │
│                ┌──────────────────┐                             │
│                │ Anomaly Detection│                             │
│                │ |X - X̂| ≥ 1.0   │                             │
│                └────────┬─────────┘                             │
│                         │                                       │
│                         ▼                                       │
│                ┌──────────────────┐                             │
│                │ Boolean DataFrame│                             │
│                │   (Anomalies)    │                             │
│                └──────────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## API Reference

### ATHENA Class

```python
class ATHENA:
    def __init__(
        self,
        model_class: type[BaseEstimator],
        model_params: dict[str, Any] | None = None,
        n_jobs: int = -1,
        verbose: int = 0
    ) -> None
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_class` | `type[BaseEstimator]` | Required | Uninstantiated scikit-learn regressor class |
| `model_params` | `dict[str, Any]` | `{}` | Parameters for model instantiation |
| `n_jobs` | `int` | `-1` | Number of parallel jobs (-1 = all cores) |
| `verbose` | `int` | `0` | Verbosity level |

### Methods

#### `fit(data, ts, interpolation_method='linear')`

Fit the ATHENA model to training data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `dict[str, pd.Series]` | Time series data with DatetimeIndex |
| `ts` | `str \| pd.Timedelta` | Target time frequency (e.g., '1h', '1D') |
| `interpolation_method` | `str` | Interpolation method: 'linear', 'time', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic' |

**Returns:** `self`

#### `predict(data, n_iterations=5, interpolation_method='linear')`

Detect anomalies in new data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `dict[str, pd.Series]` | - | Time series data to analyze |
| `n_iterations` | `int` | `5` | Number of refinement iterations |
| `interpolation_method` | `str` | `'linear'` | Interpolation method |

**Returns:** `pd.DataFrame` - Boolean DataFrame where `True` indicates anomaly

#### `get_residuals(data, n_iterations=5, interpolation_method='linear')`

Get raw residuals for custom thresholding.

**Returns:** `pd.DataFrame` - Absolute residuals in standardized units

#### `save(filepath)`

Save fitted model to disk using joblib compression.

#### `load(filepath)` (static method)

Load a saved model from disk.

## Advanced Usage

### Custom Anomaly Thresholds

```python
# Get raw residuals for custom analysis
residuals = athena.get_residuals(data, n_iterations=5)

# Custom threshold (e.g., 2 standard deviations)
custom_anomalies = residuals >= 2.0

# Percentile-based threshold
threshold = residuals.stack().quantile(0.99)
extreme_anomalies = residuals >= threshold
```

### Different Model Types

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Linear model (fast, interpretable)
athena_linear = ATHENA(
    model_class=Ridge,
    model_params={'alpha': 1.0}
)

# Gradient Boosting (high accuracy)
athena_gb = ATHENA(
    model_class=GradientBoostingRegressor,
    model_params={'n_estimators': 100, 'max_depth': 5}
)

# Neural Network (complex patterns)
athena_nn = ATHENA(
    model_class=MLPRegressor,
    model_params={'hidden_layer_sizes': (100, 50), 'max_iter': 500}
)
```

### Handling Different Time Frequencies

```python
# Hourly data resampled to daily
athena.fit(hourly_data, ts='1D', interpolation_method='cubic')

# Sub-minute data resampled to minutes
athena.fit(tick_data, ts='1min', interpolation_method='linear')

# Using Timedelta
athena.fit(data, ts=pd.Timedelta(hours=4), interpolation_method='time')
```

### Model Persistence

```python
# Save the fitted model
athena.save('models/athena_production.joblib')

# Load in production
loaded_athena = ATHENA.load('models/athena_production.joblib')
anomalies = loaded_athena.predict(new_data)
```

## Performance Considerations

### Memory Optimization

- ATHENA uses NumPy vectorization for all numerical operations
- Large datasets are processed column-wise with parallel execution
- Memory usage scales linearly with `n_samples × n_features`

### Computational Complexity

- **Training**: O(n_features × model_training_time)
- **Prediction**: O(n_iterations × n_features × model_prediction_time)

### Recommendations

| Dataset Size | Recommended Model | n_jobs |
|--------------|-------------------|--------|
| < 10K samples | RandomForest | -1 |
| 10K - 100K samples | GradientBoosting | -1 |
| > 100K samples | Ridge / LinearRegression | -1 |

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/username/athena.git
cd athena

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=athena --cov-report=html
```

### Code Quality

```bash
# Format code
black athena tests
isort athena tests

# Type checking
mypy athena

# Linting
flake8 athena tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ATHENA in your research, please cite:

```bibtex
@software{athena2024,
  title = {ATHENA: Adaptive Temporal Harmonization and Estimation for Normalized Anomalies},
  year = {2024},
  url = {https://github.com/username/athena}
}
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
