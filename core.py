"""
ATHENA: Adaptive Temporal Harmonization and Estimation for Normalized Anomalies.

A production-grade machine learning framework for time series anomaly detection
using cross-series regression and iterative refinement.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin


class ATHENA:
    """Adaptive Temporal Harmonization and Estimation for Normalized Anomalies.

    A sophisticated anomaly detection system that learns cross-series relationships
    from multiple time series and uses iterative refinement to identify anomalous
    data points. The system harmonizes time series to a common frequency, learns
    predictive models for each series based on all other series, and flags points
    where the reconstructed values deviate significantly from observed values.

    Attributes:
        model_class: The scikit-learn regressor class to use for modeling.
        model_params: Parameters to pass to the model class upon instantiation.
        n_jobs: Number of parallel jobs for fitting and prediction.
        verbose: Verbosity level for parallel operations.
        models_: Dictionary mapping column names to fitted model instances.
        scalers_: Dictionary mapping column names to (mean, std) tuples.
        ts_freq_: The time frequency used for resampling.
        feature_names_: List of column names in the fitted data.

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> athena = ATHENA(
        ...     model_class=RandomForestRegressor,
        ...     model_params={'n_estimators': 100, 'random_state': 42},
        ...     n_jobs=-1
        ... )
        >>> data = {
        ...     'sensor_a': pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-01', periods=3, freq='h')),
        ...     'sensor_b': pd.Series([2.0, 4.0, 6.0], index=pd.date_range('2024-01-01', periods=3, freq='h')),
        ... }
        >>> athena.fit(data, ts='1h', interpolation_method='linear')
        >>> anomalies = athena.predict(data, n_iterations=5)
    """

    def __init__(
        self,
        model_class: type[BaseEstimator],
        model_params: dict[str, Any] | None = None,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> None:
        """Initialize the ATHENA anomaly detector.

        Args:
            model_class: An uninstantiated scikit-learn regressor class
                (e.g., RandomForestRegressor, not an instance).
            model_params: Dictionary of parameters to pass to the model class
                upon instantiation. Defaults to empty dict.
            n_jobs: Number of parallel jobs to run during fitting and prediction.
                -1 means using all processors. Defaults to -1.
            verbose: Verbosity level for joblib parallel operations. Defaults to 0.

        Raises:
            TypeError: If model_class is an instance rather than a class.
            TypeError: If model_class is not a subclass of BaseEstimator.
        """
        # Validate model_class is a class, not an instance
        if not isinstance(model_class, type):
            raise TypeError(
                f"model_class must be an uninstantiated class, not an instance. "
                f"Got {type(model_class).__name__}. "
                f"Use RandomForestRegressor, not RandomForestRegressor()."
            )

        # Validate model_class is a scikit-learn estimator
        if not issubclass(model_class, BaseEstimator):
            raise TypeError(
                f"model_class must be a subclass of sklearn.base.BaseEstimator. "
                f"Got {model_class.__name__}."
            )

        self.model_class: type[BaseEstimator] = model_class
        self.model_params: dict[str, Any] = model_params if model_params is not None else {}
        self.n_jobs: int = n_jobs
        self.verbose: int = verbose

        # Initialize internal storage attributes
        self.models_: dict[str, BaseEstimator] = {}
        self.scalers_: dict[str, tuple[float, float]] = {}
        self.ts_freq_: str | pd.Timedelta | None = None
        self.feature_names_: list[str] = []

    def _validate_data_input(self, data: dict[str, pd.Series], context: str) -> None:
        """Validate the input data dictionary.

        Args:
            data: Dictionary of time series data to validate.
            context: Context string for error messages (e.g., 'fit', 'predict').

        Raises:
            TypeError: If data is not a dictionary.
            ValueError: If data is empty or contains non-Series values.
            TypeError: If any Series does not have a DatetimeIndex.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"data must be a dict[str, pd.Series], got {type(data).__name__}"
            )

        if len(data) == 0:
            raise ValueError(f"data dictionary cannot be empty in {context}")

        for key, series in data.items():
            if not isinstance(series, pd.Series):
                raise TypeError(
                    f"All values in data must be pd.Series. "
                    f"Key '{key}' has type {type(series).__name__}"
                )
            if not isinstance(series.index, pd.DatetimeIndex):
                raise TypeError(
                    f"All Series must have a DatetimeIndex. "
                    f"Key '{key}' has index type {type(series.index).__name__}"
                )

    def _align_and_harmonize(
        self,
        data: dict[str, pd.Series],
        ts: str | pd.Timedelta,
        interpolation_method: str,
    ) -> pd.DataFrame:
        """Align and harmonize multiple time series to a common frequency.

        Args:
            data: Dictionary of time series data.
            ts: Target time frequency for resampling (e.g., '1h', '1D').
            interpolation_method: Interpolation method for filling gaps.

        Returns:
            A harmonized DataFrame with all series aligned to the target frequency.

        Raises:
            ValueError: If the resulting DataFrame is empty after processing.
            ValueError: If interpolation_method is not valid.
        """
        valid_interpolation_methods = {
            'linear', 'time', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        }
        if interpolation_method not in valid_interpolation_methods:
            raise ValueError(
                f"interpolation_method must be one of {valid_interpolation_methods}, "
                f"got '{interpolation_method}'"
            )

        # Convert dictionary to DataFrame with outer join on indices
        df = pd.DataFrame(data)

        # Coerce all data to float64 for numerical stability
        df = df.astype(np.float64)

        # Resample to target frequency using mean aggregation
        df = df.resample(ts).mean()

        # Handle interpolation - use 'time' for DatetimeIndex when 'linear' is specified
        actual_method = interpolation_method
        if interpolation_method == 'linear' and isinstance(df.index, pd.DatetimeIndex):
            actual_method = 'time'

        # Apply interpolation to handle missing values
        df = df.interpolate(method=actual_method)

        # Drop rows that remain NaN (edges that couldn't be interpolated)
        df = df.dropna()

        # Validate resulting DataFrame is not empty
        if df.empty:
            raise ValueError(
                "The aligned and harmonized DataFrame is empty after processing. "
                "This may occur if the time series have no overlapping time range "
                "or if the resampling frequency is too coarse."
            )

        return df

    def _standardize(
        self,
        df: pd.DataFrame,
        fit_scalers: bool = True,
    ) -> pd.DataFrame:
        """Standardize the DataFrame columns using z-score normalization.

        Args:
            df: DataFrame to standardize.
            fit_scalers: If True, compute and store new scalers. If False,
                use existing scalers from self.scalers_.

        Returns:
            Standardized DataFrame with zero mean and unit variance per column.
        """
        df_scaled = df.copy()

        for col in df.columns:
            if fit_scalers:
                mean_val = df[col].mean()
                std_val = df[col].std()

                # Edge case: if std is 0, set to 1.0 to avoid division by zero
                if std_val == 0 or np.isnan(std_val):
                    std_val = 1.0

                self.scalers_[col] = (mean_val, std_val)
            else:
                if col not in self.scalers_:
                    raise ValueError(
                        f"Column '{col}' not found in fitted scalers. "
                        f"Available columns: {list(self.scalers_.keys())}"
                    )
                mean_val, std_val = self.scalers_[col]

            # Vectorized standardization: X_scaled = (X - mean) / std
            df_scaled[col] = (df[col] - mean_val) / std_val

        return df_scaled

    def _fit_single_model(
        self,
        col: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[str, BaseEstimator]:
        """Fit a single model for one target column.

        Args:
            col: Name of the target column.
            X_train: Training features (all columns except target).
            y_train: Training target values.

        Returns:
            Tuple of (column_name, fitted_model).
        """
        model = self.model_class(**self.model_params)
        model.fit(X_train, y_train)
        return (col, model)

    def fit(
        self,
        data: dict[str, pd.Series],
        ts: str | pd.Timedelta,
        interpolation_method: str = 'linear',
    ) -> ATHENA:
        """Fit the ATHENA model to the training data.

        This method aligns and harmonizes multiple time series to a common
        frequency, standardizes the data, and trains a regressor for each
        column to predict its values from all other columns.

        Args:
            data: Dictionary where keys are time series names and values are
                time-indexed Pandas Series.
            ts: Target time frequency for resampling (e.g., '1h', '1D', pd.Timedelta).
            interpolation_method: Method for interpolating missing values.
                Options: 'linear', 'time', 'nearest', 'zero', 'slinear',
                'quadratic', 'cubic'. Defaults to 'linear'.

        Returns:
            self: The fitted ATHENA instance.

        Raises:
            TypeError: If data is not properly formatted.
            ValueError: If data is empty or results in empty DataFrame.
            ValueError: If interpolation_method is invalid.
        """
        # Validate input data
        self._validate_data_input(data, context='fit')

        if len(data) < 2:
            raise ValueError(
                "ATHENA requires at least 2 time series to learn cross-series "
                f"relationships. Got {len(data)} series."
            )

        # Step 1: Data Alignment & Harmonization
        df = self._align_and_harmonize(data, ts, interpolation_method)

        # Store metadata
        self.feature_names_ = list(df.columns)
        self.ts_freq_ = ts

        # Step 2: Standardization
        df_scaled = self._standardize(df, fit_scalers=True)

        # Step 3: Model Training (Parallelized)
        # Goal: Learn f_i(X_{\i}) -> X_i for every column i
        def _prepare_and_fit(col: str) -> tuple[str, BaseEstimator]:
            """Prepare data and fit model for a single column."""
            # Define predictors (all columns except target) and target
            predictor_cols = [c for c in df_scaled.columns if c != col]
            X_train = df_scaled[predictor_cols].values
            y_train = df_scaled[col].values

            return self._fit_single_model(col, X_train, y_train)

        # Execute parallel fitting
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')(
            delayed(_prepare_and_fit)(col) for col in df_scaled.columns
        )

        # Store fitted models
        self.models_ = {col: model for col, model in results}

        return self

    def _predict_single_column(
        self,
        col: str,
        X_current: pd.DataFrame,
    ) -> tuple[str, np.ndarray]:
        """Generate predictions for a single column.

        Args:
            col: Name of the target column.
            X_current: Current state of all columns.

        Returns:
            Tuple of (column_name, predicted_values).
        """
        predictor_cols = [c for c in X_current.columns if c != col]
        X_features = X_current[predictor_cols].values
        predictions = self.models_[col].predict(X_features)
        return (col, predictions)

    def predict(
        self,
        data: dict[str, pd.Series],
        n_iterations: int = 5,
        interpolation_method: str = 'linear',
    ) -> pd.DataFrame:
        """Detect anomalies in new data using iterative refinement.

        This method processes new data through the fitted ATHENA model,
        applying iterative refinement to reconstruct expected values and
        flagging anomalies where observed values deviate significantly
        from reconstructions.

        Args:
            data: Dictionary where keys are time series names and values are
                time-indexed Pandas Series.
            n_iterations: Number of refinement iterations (the "M" parameter).
                More iterations lead to more refined reconstructions.
                Defaults to 5.
            interpolation_method: Method for interpolating missing values.
                Defaults to 'linear'.

        Returns:
            pd.DataFrame: Boolean DataFrame where True indicates an anomaly.
                Index is timestamps, columns are series names.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If input columns don't match fitted columns.
            ValueError: If n_iterations is not positive.
        """
        # Validate model is fitted
        if not self.models_:
            raise RuntimeError(
                "ATHENA model has not been fitted. Call fit() before predict()."
            )

        if n_iterations < 1:
            raise ValueError(
                f"n_iterations must be a positive integer, got {n_iterations}"
            )

        # Validate input data
        self._validate_data_input(data, context='predict')

        # Step 1: Preprocessing - replicate alignment steps from fit
        df = self._align_and_harmonize(data, self.ts_freq_, interpolation_method)

        # Strict check: ensure columns match feature_names_
        input_cols = set(df.columns)
        fitted_cols = set(self.feature_names_)

        if input_cols != fitted_cols:
            missing = fitted_cols - input_cols
            extra = input_cols - fitted_cols
            error_msg = "Column mismatch between input data and fitted model."
            if missing:
                error_msg += f" Missing columns: {missing}."
            if extra:
                error_msg += f" Extra columns: {extra}."
            raise ValueError(error_msg)

        # Reorder columns to match fitted order
        df = df[self.feature_names_]

        # Apply stored scalers for standardization
        X_input = self._standardize(df, fit_scalers=False)

        # Step 2: Iterative Refinement (The "M" Loop)
        X_current = X_input.copy()

        for iteration in range(n_iterations):
            # Parallel prediction for all columns
            results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')(
                delayed(self._predict_single_column)(col, X_current)
                for col in self.feature_names_
            )

            # Build X_next from predictions
            X_next = pd.DataFrame(
                {col: preds for col, preds in results},
                index=X_current.index,
            )

            # Reorder columns to maintain consistency
            X_next = X_next[self.feature_names_]

            # Crucial update: output of iteration k becomes input for k+1
            X_current = X_next

        # Step 3: Anomaly Detection
        # X_current is now X_final (the reconstructed matrix)
        X_final = X_current

        # Calculate absolute residuals: R = |X_input - X_final|
        residuals = np.abs(X_input.values - X_final.values)
        residuals_df = pd.DataFrame(
            residuals,
            index=X_input.index,
            columns=self.feature_names_,
        )

        # Create Boolean DataFrame for anomalies
        # True if R >= 1.0 (1 standard deviation from expected harmonic value)
        anomalies = residuals_df >= 1.0

        return anomalies

    def get_residuals(
        self,
        data: dict[str, pd.Series],
        n_iterations: int = 5,
        interpolation_method: str = 'linear',
    ) -> pd.DataFrame:
        """Get the raw residuals between observed and reconstructed values.

        This method is useful for custom anomaly thresholds or detailed analysis.

        Args:
            data: Dictionary where keys are time series names and values are
                time-indexed Pandas Series.
            n_iterations: Number of refinement iterations. Defaults to 5.
            interpolation_method: Method for interpolating missing values.
                Defaults to 'linear'.

        Returns:
            pd.DataFrame: DataFrame of absolute residuals (in standardized units).
        """
        # Validate model is fitted
        if not self.models_:
            raise RuntimeError(
                "ATHENA model has not been fitted. Call fit() before get_residuals()."
            )

        # Validate input
        self._validate_data_input(data, context='get_residuals')

        # Preprocessing
        df = self._align_and_harmonize(data, self.ts_freq_, interpolation_method)
        df = df[self.feature_names_]
        X_input = self._standardize(df, fit_scalers=False)

        # Iterative refinement
        X_current = X_input.copy()
        for _ in range(n_iterations):
            results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads')(
                delayed(self._predict_single_column)(col, X_current)
                for col in self.feature_names_
            )
            X_next = pd.DataFrame(
                {col: preds for col, preds in results},
                index=X_current.index,
            )[self.feature_names_]
            X_current = X_next

        # Calculate residuals
        residuals = np.abs(X_input.values - X_current.values)
        return pd.DataFrame(
            residuals,
            index=X_input.index,
            columns=self.feature_names_,
        )

    def save(self, filepath: str) -> None:
        """Save the fitted ATHENA model to disk.

        Args:
            filepath: Path where the model will be saved.
                Recommended extension: .joblib

        Raises:
            RuntimeError: If saving fails.
        """
        try:
            joblib.dump(self, filepath, compress=3)
        except Exception as e:
            raise RuntimeError(
                f"Failed to save ATHENA model to '{filepath}': {e}"
            ) from e

    @staticmethod
    def load(filepath: str) -> ATHENA:
        """Load a fitted ATHENA model from disk.

        Args:
            filepath: Path to the saved model file.

        Returns:
            ATHENA: The loaded ATHENA instance.

        Raises:
            RuntimeError: If loading fails.
            TypeError: If loaded object is not an ATHENA instance.
        """
        try:
            obj = joblib.load(filepath)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ATHENA model from '{filepath}': {e}"
            ) from e

        if not isinstance(obj, ATHENA):
            raise TypeError(
                f"Loaded object is not an ATHENA instance. "
                f"Got {type(obj).__name__}"
            )

        return obj

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling.

        Returns:
            Dictionary containing all instance attributes.
        """
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state when unpickling.

        Args:
            state: Dictionary containing instance attributes.
        """
        self.__dict__.update(state)

    def __repr__(self) -> str:
        """Return string representation of the ATHENA instance.

        Returns:
            String representation showing key parameters and fitted state.
        """
        fitted = bool(self.models_)
        n_features = len(self.feature_names_) if fitted else 0
        return (
            f"ATHENA("
            f"model_class={self.model_class.__name__}, "
            f"n_jobs={self.n_jobs}, "
            f"fitted={fitted}, "
            f"n_features={n_features})"
        )
