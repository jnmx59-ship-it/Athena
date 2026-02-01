#!/usr/bin/env python
"""
Example: Basic ATHENA Usage

This example demonstrates:
1. Creating synthetic time series data
2. Injecting anomalies
3. Fitting the ATHENA model
4. Detecting anomalies
5. Analyzing results
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from athena import ATHENA


def generate_synthetic_data(
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, pd.Series]:
    """Generate synthetic time series data with correlations."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="h")
    
    # Base signal (shared component)
    base = np.sin(np.linspace(0, 20 * np.pi, n_samples))
    
    # Create correlated series with different characteristics
    data = {
        "temperature": pd.Series(
            base * 10 + 20 + np.random.normal(0, 0.5, n_samples),
            index=dates,
            name="temperature",
        ),
        "humidity": pd.Series(
            -base * 15 + 60 + np.random.normal(0, 1, n_samples),
            index=dates,
            name="humidity",
        ),
        "pressure": pd.Series(
            base * 3 + np.sin(np.linspace(0, 10 * np.pi, n_samples)) * 2
            + 1013 + np.random.normal(0, 0.3, n_samples),
            index=dates,
            name="pressure",
        ),
        "wind_speed": pd.Series(
            np.abs(base * 5) + 10 + np.random.normal(0, 0.8, n_samples),
            index=dates,
            name="wind_speed",
        ),
    }
    
    return data


def inject_anomalies(
    data: dict[str, pd.Series],
) -> tuple[dict[str, pd.Series], dict[str, list[int]]]:
    """Inject known anomalies into the data."""
    data = {k: v.copy() for k, v in data.items()}
    injected_locations = {}
    
    # Temperature spike (heat wave)
    data["temperature"].iloc[200:210] += 15
    injected_locations["temperature"] = list(range(200, 210))
    
    # Humidity drop
    data["humidity"].iloc[500:515] -= 25
    injected_locations["humidity"] = list(range(500, 515))
    
    # Pressure anomaly
    data["pressure"].iloc[700:705] += 10
    injected_locations["pressure"] = list(range(700, 705))
    
    return data, injected_locations


def main():
    """Run the example."""
    print("=" * 60)
    print("ATHENA Example: Anomaly Detection in Multivariate Time Series")
    print("=" * 60)
    
    # Step 1: Generate data
    print("\n1. Generating synthetic time series data...")
    data = generate_synthetic_data(n_samples=1000)
    print(f"   Created {len(data)} time series with {len(next(iter(data.values())))} samples each")
    
    # Step 2: Inject anomalies
    print("\n2. Injecting known anomalies...")
    anomalous_data, injection_locations = inject_anomalies(data)
    for series, locs in injection_locations.items():
        print(f"   - {series}: indices {locs[0]}-{locs[-1]}")
    
    # Step 3: Initialize and fit ATHENA
    print("\n3. Initializing ATHENA model...")
    athena = ATHENA(
        model_class=RandomForestRegressor,
        model_params={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": 1,  # Use 1 job within model since ATHENA parallelizes over features
        },
        n_jobs=-1,  # Use all cores for parallel feature processing
        verbose=0,
    )
    
    print("\n4. Fitting model on clean data...")
    athena.fit(data, ts="1h", interpolation_method="linear")
    print(f"   Fitted models for {len(athena.feature_names_)} features")
    print(f"   Features: {athena.feature_names_}")
    
    # Step 4: Detect anomalies
    print("\n5. Detecting anomalies in data with injected anomalies...")
    anomalies = athena.predict(anomalous_data, n_iterations=5)
    
    # Step 5: Analyze results
    print("\n6. Results:")
    print("-" * 40)
    
    total_anomalies = anomalies.sum().sum()
    print(f"   Total anomalies detected: {total_anomalies}")
    
    print("\n   Anomalies by series:")
    for col in anomalies.columns:
        count = anomalies[col].sum()
        print(f"   - {col}: {count} anomalous points")
    
    # Check detection of injected anomalies
    print("\n7. Detection of injected anomalies:")
    print("-" * 40)
    
    for series, locs in injection_locations.items():
        detected = anomalies.iloc[locs][series].sum()
        total_injected = len(locs)
        pct = (detected / total_injected) * 100
        print(f"   - {series}: {detected}/{total_injected} detected ({pct:.1f}%)")
    
    # Get residuals for custom analysis
    print("\n8. Getting raw residuals for custom thresholding...")
    residuals = athena.get_residuals(anomalous_data, n_iterations=5)
    
    print("\n   Residual statistics:")
    print(residuals.describe().round(3))
    
    # Save and load example
    print("\n9. Demonstrating model persistence...")
    model_path = "/tmp/athena_example_model.joblib"
    athena.save(model_path)
    print(f"   Model saved to: {model_path}")
    
    loaded_athena = ATHENA.load(model_path)
    print(f"   Model loaded successfully: {repr(loaded_athena)}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
