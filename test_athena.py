"""
Comprehensive test suite for the ATHENA anomaly detection framework.

Tests cover:
- Initialization and validation
- Data harmonization and alignment
- Model fitting
- Anomaly prediction
- Model persistence
- Edge cases and error handling
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from athena import ATHENA


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_data() -> dict[str, pd.Series]:
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="h")
    
    return {
        "series_a": pd.Series(
            np.sin(np.linspace(0, 4 * np.pi, 200)) * 10 + np.random.normal(0, 0.5, 200),
            index=dates,
        ),
        "series_b": pd.Series(
            np.cos(np.linspace(0, 4 * np.pi, 200)) * 8 + np.random.normal(0, 0.5, 200),
            index=dates,
        ),
        "series_c": pd.Series(
            np.sin(np.linspace(0, 2 * np.pi, 200)) * 5 + 15 + np.random.normal(0, 0.3, 200),
            index=dates,
        ),
    }


@pytest.fixture
def sample_data_with_anomalies(sample_data: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Create sample data with injected anomalies."""
    data = {k: v.copy() for k, v in sample_data.items()}
    # Inject anomalies
    data["series_a"].iloc[50:55] += 30  # Large spike
    data["series_b"].iloc[100:105] -= 25  # Large drop
    return data


@pytest.fixture
def misaligned_data() -> dict[str, pd.Series]:
    """Create time series with different start/end times and frequencies."""
    np.random.seed(42)
    
    dates_a = pd.date_range("2024-01-01 00:00", periods=100, freq="h")
    dates_b = pd.date_range("2024-01-01 06:00", periods=80, freq="h")
    dates_c = pd.date_range("2024-01-01 00:00", periods=50, freq="2h")
    
    return {
        "series_a": pd.Series(np.random.randn(100) * 2 + 10, index=dates_a),
        "series_b": pd.Series(np.random.randn(80) * 3 + 20, index=dates_b),
        "series_c": pd.Series(np.random.randn(50) * 1.5 + 15, index=dates_c),
    }


@pytest.fixture
def fitted_athena(sample_data: dict[str, pd.Series]) -> ATHENA:
    """Create a fitted ATHENA instance."""
    athena = ATHENA(
        model_class=DecisionTreeRegressor,
        model_params={"max_depth": 5, "random_state": 42},
        n_jobs=1,
    )
    athena.fit(sample_data, ts="1h", interpolation_method="linear")
    return athena


# ============================================================================
# Initialization Tests
# ============================================================================


class TestATHENAInitialization:
    """Tests for ATHENA initialization."""

    def test_valid_initialization(self):
        """Test successful initialization with valid parameters."""
        athena = ATHENA(
            model_class=RandomForestRegressor,
            model_params={"n_estimators": 10, "random_state": 42},
            n_jobs=2,
            verbose=1,
        )
        
        assert athena.model_class == RandomForestRegressor
        assert athena.model_params == {"n_estimators": 10, "random_state": 42}
        assert athena.n_jobs == 2
        assert athena.verbose == 1
        assert athena.models_ == {}
        assert athena.scalers_ == {}
        assert athena.ts_freq_ is None
        assert athena.feature_names_ == []

    def test_default_parameters(self):
        """Test initialization with default parameters."""
        athena = ATHENA(model_class=LinearRegression)
        
        assert athena.model_params == {}
        assert athena.n_jobs == -1
        assert athena.verbose == 0

    def test_initialization_with_instance_raises_error(self):
        """Test that passing an instance instead of class raises TypeError."""
        model_instance = RandomForestRegressor(n_estimators=10)
        
        with pytest.raises(TypeError, match="must be an uninstantiated class"):
            ATHENA(model_class=model_instance)

    def test_initialization_with_non_estimator_raises_error(self):
        """Test that passing non-estimator class raises TypeError."""
        with pytest.raises(TypeError, match="must be a subclass of sklearn.base.BaseEstimator"):
            ATHENA(model_class=str)

    def test_repr(self):
        """Test string representation."""
        athena = ATHENA(model_class=Ridge, n_jobs=4)
        repr_str = repr(athena)
        
        assert "Ridge" in repr_str
        assert "n_jobs=4" in repr_str
        assert "fitted=False" in repr_str


# ============================================================================
# Data Validation Tests
# ============================================================================


class TestDataValidation:
    """Tests for input data validation."""

    def test_invalid_data_type(self):
        """Test error handling for non-dict input."""
        athena = ATHENA(model_class=LinearRegression)
        
        with pytest.raises(TypeError, match="must be a dict"):
            athena.fit("not a dict", ts="1h")

    def test_empty_data_dict(self):
        """Test error handling for empty data dictionary."""
        athena = ATHENA(model_class=LinearRegression)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            athena.fit({}, ts="1h")

    def test_non_series_values(self):
        """Test error handling for non-Series values in dict."""
        athena = ATHENA(model_class=LinearRegression)
        data = {"series_a": [1, 2, 3]}  # List instead of Series
        
        with pytest.raises(TypeError, match="must be pd.Series"):
            athena.fit(data, ts="1h")

    def test_non_datetime_index(self):
        """Test error handling for non-DatetimeIndex."""
        athena = ATHENA(model_class=LinearRegression)
        data = {"series_a": pd.Series([1, 2, 3], index=[0, 1, 2])}
        
        with pytest.raises(TypeError, match="must have a DatetimeIndex"):
            athena.fit(data, ts="1h")

    def test_single_series_raises_error(self):
        """Test that single series raises error (need >= 2 for cross-series)."""
        athena = ATHENA(model_class=LinearRegression)
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        data = {"single": pd.Series(np.random.randn(10), index=dates)}
        
        with pytest.raises(ValueError, match="requires at least 2 time series"):
            athena.fit(data, ts="1h")

    def test_invalid_interpolation_method(self, sample_data):
        """Test error handling for invalid interpolation method."""
        athena = ATHENA(model_class=LinearRegression)
        
        with pytest.raises(ValueError, match="interpolation_method must be one of"):
            athena.fit(sample_data, ts="1h", interpolation_method="invalid")


# ============================================================================
# Fit Tests
# ============================================================================


class TestATHENAFit:
    """Tests for the fit method."""

    def test_basic_fit(self, sample_data):
        """Test basic fitting functionality."""
        athena = ATHENA(
            model_class=DecisionTreeRegressor,
            model_params={"max_depth": 3, "random_state": 42},
            n_jobs=1,
        )
        
        result = athena.fit(sample_data, ts="1h", interpolation_method="linear")
        
        # Check return value
        assert result is athena
        
        # Check models were trained
        assert len(athena.models_) == 3
        assert set(athena.models_.keys()) == {"series_a", "series_b", "series_c"}
        
        # Check scalers were computed
        assert len(athena.scalers_) == 3
        for mean, std in athena.scalers_.values():
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std > 0
        
        # Check metadata
        assert athena.ts_freq_ == "1h"
        assert athena.feature_names_ == list(sample_data.keys())

    def test_fit_with_different_frequencies(self, sample_data):
        """Test fitting with different target frequencies."""
        athena = ATHENA(model_class=LinearRegression, n_jobs=1)
        
        # Hourly to daily
        athena.fit(sample_data, ts="1D", interpolation_method="linear")
        assert athena.ts_freq_ == "1D"

    def test_fit_with_misaligned_data(self, misaligned_data):
        """Test fitting with misaligned time series."""
        athena = ATHENA(
            model_class=DecisionTreeRegressor,
            model_params={"max_depth": 3},
            n_jobs=1,
        )
        
        athena.fit(misaligned_data, ts="1h", interpolation_method="linear")
        
        assert len(athena.models_) == 3
        assert len(athena.feature_names_) == 3

    def test_fit_handles_zero_std(self):
        """Test that fit handles zero standard deviation correctly."""
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        data = {
            "constant": pd.Series([5.0] * 50, index=dates),  # Zero std
            "varying": pd.Series(np.random.randn(50), index=dates),
        }
        
        athena = ATHENA(model_class=LinearRegression, n_jobs=1)
        athena.fit(data, ts="1h")
        
        # Std should be set to 1.0 for constant series
        mean, std = athena.scalers_["constant"]
        assert std == 1.0
        assert mean == 5.0

    def test_fit_interpolation_methods(self, sample_data):
        """Test different interpolation methods."""
        methods = ["linear", "time", "nearest", "zero", "slinear", "quadratic", "cubic"]
        
        for method in methods:
            athena = ATHENA(model_class=LinearRegression, n_jobs=1)
            athena.fit(sample_data, ts="1h", interpolation_method=method)
            assert len(athena.models_) == 3

    def test_fit_with_timedelta(self, sample_data):
        """Test fitting with pd.Timedelta frequency."""
        athena = ATHENA(model_class=LinearRegression, n_jobs=1)
        athena.fit(sample_data, ts=pd.Timedelta(hours=2))
        
        assert athena.ts_freq_ == pd.Timedelta(hours=2)


# ============================================================================
# Predict Tests
# ============================================================================


class TestATHENAPredict:
    """Tests for the predict method."""

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predict before fit raises RuntimeError."""
        athena = ATHENA(model_class=LinearRegression)
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            athena.predict(sample_data)

    def test_basic_predict(self, fitted_athena, sample_data):
        """Test basic prediction functionality."""
        anomalies = fitted_athena.predict(sample_data, n_iterations=3)
        
        # Check output type and shape
        assert isinstance(anomalies, pd.DataFrame)
        assert anomalies.dtypes.apply(lambda x: x == bool).all()
        assert list(anomalies.columns) == list(sample_data.keys())

    def test_predict_detects_anomalies(self, sample_data):
        """Test that predict detects injected anomalies."""
        athena = ATHENA(
            model_class=RandomForestRegressor,
            model_params={"n_estimators": 50, "random_state": 42},
            n_jobs=1,
        )
        athena.fit(sample_data, ts="1h")
        
        # Create data with clear anomalies
        anomalous_data = {k: v.copy() for k, v in sample_data.items()}
        anomalous_data["series_a"].iloc[100:105] += 50  # Large spike
        
        anomalies = athena.predict(anomalous_data, n_iterations=5)
        
        # Should detect some anomalies in the spike region
        assert anomalies.loc[anomalies.index[100:105], "series_a"].sum() > 0

    def test_predict_with_different_iterations(self, fitted_athena, sample_data):
        """Test prediction with different iteration counts."""
        for n_iter in [1, 3, 5, 10]:
            anomalies = fitted_athena.predict(sample_data, n_iterations=n_iter)
            assert isinstance(anomalies, pd.DataFrame)

    def test_predict_invalid_iterations(self, fitted_athena, sample_data):
        """Test that invalid n_iterations raises error."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            fitted_athena.predict(sample_data, n_iterations=0)
        
        with pytest.raises(ValueError, match="must be a positive integer"):
            fitted_athena.predict(sample_data, n_iterations=-1)

    def test_predict_column_mismatch(self, fitted_athena):
        """Test error handling for column mismatch."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        wrong_data = {
            "wrong_col": pd.Series(np.random.randn(100), index=dates),
            "series_b": pd.Series(np.random.randn(100), index=dates),
            "series_c": pd.Series(np.random.randn(100), index=dates),
        }
        
        with pytest.raises(ValueError, match="Column mismatch"):
            fitted_athena.predict(wrong_data)

    def test_get_residuals(self, fitted_athena, sample_data):
        """Test get_residuals method."""
        residuals = fitted_athena.get_residuals(sample_data, n_iterations=3)
        
        assert isinstance(residuals, pd.DataFrame)
        assert (residuals >= 0).all().all()  # Absolute residuals are non-negative
        assert list(residuals.columns) == list(sample_data.keys())


# ============================================================================
# Persistence Tests
# ============================================================================


class TestATHENAPersistence:
    """Tests for model save/load functionality."""

    def test_save_and_load(self, fitted_athena, sample_data):
        """Test saving and loading a fitted model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "athena_model.joblib"
            
            # Save
            fitted_athena.save(str(filepath))
            assert filepath.exists()
            
            # Load
            loaded = ATHENA.load(str(filepath))
            
            # Verify loaded model
            assert isinstance(loaded, ATHENA)
            assert loaded.feature_names_ == fitted_athena.feature_names_
            assert loaded.ts_freq_ == fitted_athena.ts_freq_
            assert set(loaded.models_.keys()) == set(fitted_athena.models_.keys())
            
            # Verify predictions match
            orig_pred = fitted_athena.predict(sample_data, n_iterations=2)
            loaded_pred = loaded.predict(sample_data, n_iterations=2)
            
            pd.testing.assert_frame_equal(orig_pred, loaded_pred)

    def test_load_invalid_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(RuntimeError, match="Failed to load"):
            ATHENA.load("/nonexistent/path/model.joblib")

    def test_load_wrong_type(self):
        """Test loading wrong object type raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import joblib
            filepath = Path(tmpdir) / "wrong_type.joblib"
            joblib.dump({"not": "athena"}, filepath)
            
            with pytest.raises(TypeError, match="not an ATHENA instance"):
                ATHENA.load(str(filepath))

    def test_getstate_setstate(self, fitted_athena):
        """Test __getstate__ and __setstate__ methods."""
        state = fitted_athena.__getstate__()
        
        assert isinstance(state, dict)
        assert "models_" in state
        assert "scalers_" in state
        
        # Create new instance and restore state
        new_athena = object.__new__(ATHENA)
        new_athena.__setstate__(state)
        
        assert new_athena.feature_names_ == fitted_athena.feature_names_


# ============================================================================
# Integration Tests
# ============================================================================


class TestATHENAIntegration:
    """Integration tests for full workflows."""

    def test_full_workflow(self):
        """Test complete workflow from data creation to anomaly detection."""
        # Create realistic data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=500, freq="h")
        
        data = {
            "temperature": pd.Series(
                20 + 10 * np.sin(np.linspace(0, 10 * np.pi, 500))
                + np.random.normal(0, 1, 500),
                index=dates,
            ),
            "pressure": pd.Series(
                1013 + 5 * np.cos(np.linspace(0, 10 * np.pi, 500))
                + np.random.normal(0, 0.5, 500),
                index=dates,
            ),
            "humidity": pd.Series(
                60 + 15 * np.sin(np.linspace(0, 5 * np.pi, 500))
                + np.random.normal(0, 2, 500),
                index=dates,
            ),
        }
        
        # Inject anomalies
        data["temperature"].iloc[200:210] += 20  # Heat wave
        data["pressure"].iloc[350:360] -= 15  # Pressure drop
        
        # Fit model
        athena = ATHENA(
            model_class=RandomForestRegressor,
            model_params={"n_estimators": 100, "max_depth": 10, "random_state": 42},
            n_jobs=-1,
        )
        athena.fit(data, ts="1h", interpolation_method="linear")
        
        # Predict anomalies
        anomalies = athena.predict(data, n_iterations=5)
        
        # Verify structure
        assert isinstance(anomalies, pd.DataFrame)
        assert anomalies.shape[1] == 3
        
        # Verify some anomalies were detected in injected regions
        temp_anomalies = anomalies.loc[anomalies.index[200:210], "temperature"].sum()
        pressure_anomalies = anomalies.loc[anomalies.index[350:360], "pressure"].sum()
        
        # At least some should be detected
        assert temp_anomalies + pressure_anomalies > 0

    def test_parallel_vs_sequential(self, sample_data):
        """Test that parallel and sequential execution give same results."""
        athena_parallel = ATHENA(
            model_class=DecisionTreeRegressor,
            model_params={"max_depth": 5, "random_state": 42},
            n_jobs=-1,
        )
        athena_sequential = ATHENA(
            model_class=DecisionTreeRegressor,
            model_params={"max_depth": 5, "random_state": 42},
            n_jobs=1,
        )
        
        athena_parallel.fit(sample_data, ts="1h")
        athena_sequential.fit(sample_data, ts="1h")
        
        pred_parallel = athena_parallel.predict(sample_data, n_iterations=3)
        pred_sequential = athena_sequential.predict(sample_data, n_iterations=3)
        
        pd.testing.assert_frame_equal(pred_parallel, pred_sequential)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimal_data(self):
        """Test with minimal viable data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        data = {
            "a": pd.Series(np.random.randn(10), index=dates),
            "b": pd.Series(np.random.randn(10), index=dates),
        }
        
        athena = ATHENA(model_class=LinearRegression, n_jobs=1)
        athena.fit(data, ts="1h")
        anomalies = athena.predict(data, n_iterations=1)
        
        assert anomalies.shape == (10, 2)

    def test_large_number_of_series(self):
        """Test with many series."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        data = {
            f"series_{i}": pd.Series(np.random.randn(50), index=dates)
            for i in range(20)
        }
        
        athena = ATHENA(
            model_class=LinearRegression,
            n_jobs=-1,
        )
        athena.fit(data, ts="1h")
        
        assert len(athena.models_) == 20
        
        anomalies = athena.predict(data, n_iterations=2)
        assert anomalies.shape[1] == 20

    def test_high_correlation_series(self):
        """Test with highly correlated series."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        base = np.random.randn(100)
        
        data = {
            "base": pd.Series(base, index=dates),
            "corr1": pd.Series(base * 2 + 1, index=dates),
            "corr2": pd.Series(base * 0.5 - 2, index=dates),
        }
        
        athena = ATHENA(model_class=LinearRegression, n_jobs=1)
        athena.fit(data, ts="1h")
        anomalies = athena.predict(data, n_iterations=3)
        
        # Highly correlated data should have few anomalies
        total_anomalies = anomalies.sum().sum()
        assert total_anomalies < 50  # Less than half flagged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
