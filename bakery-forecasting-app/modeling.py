"""Reproducible modeling utilities for the bakery forecasting app.

The chronology is deliberate:
    2013-2015  fit candidate models
    2016       compare candidates and lock the final specification
    2013-2016  refit the locked baseline and final models
    2017       evaluate once and display forecasts in the app

Keeping these steps in one module makes the analysis testable and prevents the
2017 test period from leaking into model selection.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REQUIRED_COLUMNS = {
    "date",
    "store_id",
    "item_id",
    "sales",
    "store_type",
    "bad_weather",
    "promotion",
    "day",
    "month",
}

BASELINE_FORMULA = "sales ~ promotion"

# The reference categories make coefficients and diagnostics reproducible.
FINAL_FORMULA = (
    "np.log1p(sales) ~ promotion + bad_weather + "
    "C(store_id, Treatment(reference=1)) + "
    'C(day, Treatment(reference="Monday"))'
)

CURVED_PROMOTION_FORMULA = FINAL_FORMULA + " + I(promotion ** 2)"

# If validation RMSE differs by no more than 0.1%, treat the models as a
# practical tie and select the model with fewer estimated parameters.
PARSIMONY_TOLERANCE = 0.001


# Candidate models are intentionally modest in size. The final specification is
# chosen on 2016 validation performance and simplicity, never on 2017 results.
CANDIDATE_MODELS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    {
        "Promotion only (baseline)": {
            "formula": BASELINE_FORMULA,
            "log_response": False,
        },
        "Operational additive (levels)": {
            "formula": (
                "sales ~ promotion + bad_weather + "
                "C(store_id, Treatment(reference=1)) + "
                'C(day, Treatment(reference="Monday"))'
            ),
            "log_response": False,
        },
        "Operational log": {
            "formula": FINAL_FORMULA,
            "log_response": True,
        },
        "Operational log + curved promotion": {
            "formula": CURVED_PROMOTION_FORMULA,
            "log_response": True,
        },
        "Log + curved promotion + month": {
            "formula": CURVED_PROMOTION_FORMULA
            + " + C(month, Treatment(reference=1))",
            "log_response": True,
        },
        "Log + promotion by store type": {
            "formula": CURVED_PROMOTION_FORMULA + " + promotion:C(store_type)",
            "log_response": True,
        },
    }
)

FINAL_MODEL_NAME = "Operational log"
BASELINE_MODEL_NAME = "Promotion only (baseline)"


def load_and_validate_data(path: str | Path) -> pd.DataFrame:
    """Load the CSV, validate its structure, and add a calendar year column."""

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Bakery data was not found at {data_path}")

    data = pd.read_csv(data_path, parse_dates=["date"])
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("The bakery data contains missing values in required fields")

    data = data.copy()
    expected_month_names = data["date"].dt.month_name()
    if not data["month"].astype(str).eq(expected_month_names).all():
        raise ValueError("The supplied month labels do not match the date field")
    if not data["day"].astype(str).eq(data["date"].dt.day_name()).all():
        raise ValueError("The supplied weekday labels do not match the date field")
    data["store_id"] = data["store_id"].astype(int)
    # The supplied CSV stores month names; use the parsed date for a dependable
    # numeric month that supports filtering and categorical model terms.
    data["month"] = data["date"].dt.month.astype(int)
    data["bad_weather"] = data["bad_weather"].astype(int)
    data["year"] = data["date"].dt.year.astype(int)

    if data.duplicated(["date", "store_id"]).any():
        raise ValueError("Each store-date pair must be unique")
    if set(data["year"].unique()) != {2013, 2014, 2015, 2016, 2017}:
        raise ValueError("Expected complete calendar years from 2013 through 2017")
    if set(data["store_id"].unique()) != set(range(1, 11)):
        raise ValueError("Expected store IDs 1 through 10")
    if set(data["item_id"].unique()) != {1032943}:
        raise ValueError("Expected all rows to describe bakery item 1032943")
    if data.groupby("store_id")["store_type"].nunique().gt(1).any():
        raise ValueError("Each store must have one stable store-type classification")
    if not data.groupby("store_id").size().eq(1_826).all():
        raise ValueError("Expected a complete 1,826-day history for every store")
    if not data.groupby("date").size().eq(10).all():
        raise ValueError("Expected one observation per store on every date")
    if not data["bad_weather"].isin([0, 1]).all():
        raise ValueError("bad_weather must contain only 0 and 1")
    if (data["sales"] < 0).any():
        raise ValueError("Sales cannot be negative")
    if not data["promotion"].between(0, 1).all():
        raise ValueError("Promotion must be a proportion between 0 and 1")

    return data.sort_values(["date", "store_id"]).reset_index(drop=True)


def prepare_splits(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create chronological model-development, validation, and test splits."""

    splits = {
        "development": data[data["year"].between(2013, 2015)].copy(),
        "validation": data[data["year"] == 2016].copy(),
        "final_training": data[data["year"].between(2013, 2016)].copy(),
        "test": data[data["year"] == 2017].copy(),
    }
    if any(frame.empty for frame in splits.values()):
        raise ValueError("One or more chronological data splits are empty")
    if set(splits["final_training"]["year"]).intersection(
        set(splits["test"]["year"])
    ):
        raise ValueError("The 2017 test period leaked into final model training")
    return splits


def regression_metrics(actual: pd.Series | np.ndarray, predicted: Any) -> dict[str, float]:
    """Return RMSE, MAE, and signed forecast bias on the sales scale."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("Actual and predicted values must have the same shape")
    errors = predicted_array - actual_array
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
    }


def fit_model(
    data: pd.DataFrame, formula: str, log_response: bool
) -> dict[str, Any]:
    """Fit an OLS model and estimate a Duan smearing correction when needed."""

    result = smf.ols(formula=formula, data=data).fit()
    smearing_factor = float(np.mean(np.exp(result.resid))) if log_response else 1.0
    return {
        "result": result,
        "log_response": log_response,
        "smearing_factor": smearing_factor,
    }


def predict_sales(model_bundle: dict[str, Any], data: pd.DataFrame) -> np.ndarray:
    """Predict nonnegative sales, returning log models to the original scale."""

    raw_prediction = np.asarray(model_bundle["result"].predict(data), dtype=float)
    if model_bundle["log_response"]:
        prediction = (
            np.exp(raw_prediction) * model_bundle["smearing_factor"] - 1.0
        )
    else:
        prediction = raw_prediction
    return np.maximum(0.0, prediction)


def compare_candidates(
    development: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Score candidates on 2016 and apply the documented parsimony rule."""

    rows: list[dict[str, Any]] = []
    errors_by_model: dict[str, np.ndarray] = {}

    for model_name, specification in CANDIDATE_MODELS.items():
        fitted = fit_model(
            development,
            specification["formula"],
            specification["log_response"],
        )
        predictions = predict_sales(fitted, validation)
        metrics = regression_metrics(validation["sales"], predictions)
        errors_by_model[model_name] = validation["sales"].to_numpy() - predictions
        rows.append(
            {
                "model": model_name,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "bias": metrics["bias"],
                "parameters": int(fitted["result"].df_model + 1),
            }
        )

    comparison = pd.DataFrame(rows)
    best_rmse = comparison["rmse"].min()
    practically_tied = comparison[
        comparison["rmse"] <= best_rmse * (1.0 + PARSIMONY_TOLERANCE)
    ]
    selected_name = practically_tied.sort_values(
        ["parameters", "rmse"], ascending=[True, True]
    ).iloc[0]["model"]
    if selected_name != FINAL_MODEL_NAME:
        raise RuntimeError(
            f"Selection rule chose {selected_name!r}, not configured final model "
            f"{FINAL_MODEL_NAME!r}"
        )
    comparison["selected"] = comparison["model"].eq(selected_name)
    comparison = comparison.sort_values("rmse").reset_index(drop=True)
    return comparison, errors_by_model[selected_name]


def _modeled_driver_changes(final_model: dict[str, Any]) -> pd.DataFrame:
    """Translate key model terms into approachable percentage changes."""

    reference = pd.DataFrame(
        {
            "store_id": [1],
            "day": ["Monday"],
            "bad_weather": [0],
            "promotion": [0.0],
            "store_type": ["urban"],
            "month": [1],
        }
    )
    base = float(predict_sales(final_model, reference)[0])

    scenarios = [
        ("10-point promotion", {"promotion": 0.10}, "0% to 10% promotion"),
        ("Bad weather", {"bad_weather": 1}, "bad vs normal weather"),
        ("Saturday", {"day": "Saturday"}, "Saturday vs Monday"),
        ("Sunday", {"day": "Sunday"}, "Sunday vs Monday"),
    ]
    rows = []
    for driver, changes, comparison in scenarios:
        scenario = reference.copy()
        for field, value in changes.items():
            scenario[field] = value
        scenario_prediction = float(predict_sales(final_model, scenario)[0])
        rows.append(
            {
                "driver": driver,
                "modeled_change_pct": 100.0 * (scenario_prediction / base - 1.0),
                "comparison": comparison,
            }
        )
    return pd.DataFrame(rows)


def run_full_analysis(data: pd.DataFrame) -> dict[str, Any]:
    """Run model selection, refit locked models, and evaluate 2017 forecasts."""

    splits = prepare_splits(data)
    comparison, validation_errors = compare_candidates(
        splits["development"], splits["validation"]
    )

    # Use held-out 2016 residuals for an empirical 80% forecast band. This band
    # is fixed before 2017 is evaluated.
    error_q10, error_q90 = np.quantile(validation_errors, [0.10, 0.90])

    baseline_model = fit_model(
        splits["final_training"], BASELINE_FORMULA, log_response=False
    )
    final_model = fit_model(
        splits["final_training"], FINAL_FORMULA, log_response=True
    )

    forecasts = splits["test"].copy()
    forecasts["baseline_forecast"] = predict_sales(baseline_model, forecasts)
    forecasts["final_forecast"] = predict_sales(final_model, forecasts)
    forecasts["interval_lower"] = np.maximum(
        0.0, forecasts["final_forecast"] + error_q10
    )
    forecasts["interval_upper"] = np.maximum(
        forecasts["interval_lower"], forecasts["final_forecast"] + error_q90
    )
    forecasts["inside_interval"] = forecasts["sales"].between(
        forecasts["interval_lower"], forecasts["interval_upper"], inclusive="both"
    )

    test_rows = []
    for model_name, prediction_column in [
        (BASELINE_MODEL_NAME, "baseline_forecast"),
        (FINAL_MODEL_NAME, "final_forecast"),
    ]:
        metrics = regression_metrics(forecasts["sales"], forecasts[prediction_column])
        test_rows.append({"model": model_name, **metrics})
    test_comparison = pd.DataFrame(test_rows)

    fitted_log = np.asarray(final_model["result"].fittedvalues, dtype=float)
    residual_log = np.asarray(final_model["result"].resid, dtype=float)
    diagnostics = pd.DataFrame(
        {
            "fitted_log_sales": fitted_log,
            "residual_log_sales": residual_log,
        }
    )

    baseline_validation = comparison.loc[
        comparison["model"] == BASELINE_MODEL_NAME
    ].iloc[0]
    final_validation = comparison.loc[
        comparison["model"] == FINAL_MODEL_NAME
    ].iloc[0]
    baseline_test = test_comparison.loc[
        test_comparison["model"] == BASELINE_MODEL_NAME
    ].iloc[0]
    final_test = test_comparison.loc[
        test_comparison["model"] == FINAL_MODEL_NAME
    ].iloc[0]

    training_residuals = splits["final_training"][["store_id", "date"]].copy()
    training_residuals["residual"] = residual_log
    lag_one_by_store = (
        training_residuals.sort_values(["store_id", "date"])
        .groupby("store_id")["residual"]
        .apply(lambda residual: residual.autocorr(lag=1))
    )
    absolute_residual_fitted_correlation = float(
        np.corrcoef(np.abs(residual_log), fitted_log)[0, 1]
    )

    summary = {
        "development_rows": int(len(splits["development"])),
        "validation_rows": int(len(splits["validation"])),
        "final_training_rows": int(len(splits["final_training"])),
        "test_rows": int(len(splits["test"])),
        "validation_rmse_improvement_pct": float(
            100.0 * (1.0 - final_validation["rmse"] / baseline_validation["rmse"])
        ),
        "validation_mae_improvement_pct": float(
            100.0 * (1.0 - final_validation["mae"] / baseline_validation["mae"])
        ),
        "test_rmse_improvement_pct": float(
            100.0 * (1.0 - final_test["rmse"] / baseline_test["rmse"])
        ),
        "test_mae_improvement_pct": float(
            100.0 * (1.0 - final_test["mae"] / baseline_test["mae"])
        ),
        "residual_skew_log_scale": float(pd.Series(residual_log).skew()),
        "absolute_residual_fitted_correlation": absolute_residual_fitted_correlation,
        "mean_store_lag1_residual_correlation": float(lag_one_by_store.mean()),
        "smearing_factor": float(final_model["smearing_factor"]),
        "interval_error_q10": float(error_q10),
        "interval_error_q90": float(error_q90),
        "test_interval_coverage_pct": float(100.0 * forecasts["inside_interval"].mean()),
        "final_formula": FINAL_FORMULA,
    }

    return {
        "forecasts": forecasts.reset_index(drop=True),
        "validation_comparison": comparison,
        "test_comparison": test_comparison,
        "diagnostics": diagnostics,
        "driver_changes": _modeled_driver_changes(final_model),
        "summary": summary,
    }

