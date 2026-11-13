"""Tests for data integrity, split isolation, metrics, and forecasts."""

from pathlib import Path

import numpy as np

from modeling import (
    BASELINE_MODEL_NAME,
    FINAL_MODEL_NAME,
    load_and_validate_data,
    prepare_splits,
    regression_metrics,
    run_full_analysis,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bakerysales.csv"


def test_source_data_is_complete_daily_panel() -> None:
    data = load_and_validate_data(DATA_PATH)
    assert len(data) == 18_260
    assert data["store_id"].nunique() == 10
    assert data.groupby("store_id").size().eq(1_826).all()
    assert data[["date", "store_id"]].duplicated().sum() == 0


def test_chronological_splits_keep_2017_isolated() -> None:
    data = load_and_validate_data(DATA_PATH)
    splits = prepare_splits(data)
    assert set(splits["development"]["year"]) == {2013, 2014, 2015}
    assert set(splits["validation"]["year"]) == {2016}
    assert set(splits["final_training"]["year"]) == {2013, 2014, 2015, 2016}
    assert set(splits["test"]["year"]) == {2017}
    assert splits["final_training"]["date"].max() < splits["test"]["date"].min()


def test_regression_metrics_on_known_values() -> None:
    metrics = regression_metrics(
        np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0])
    )
    assert np.isclose(metrics["rmse"], np.sqrt(17.0 / 3.0))
    assert np.isclose(metrics["mae"], 7.0 / 3.0)
    assert np.isclose(metrics["bias"], 1.0)


def test_full_analysis_produces_valid_2017_forecasts() -> None:
    data = load_and_validate_data(DATA_PATH)
    analysis = run_full_analysis(data)
    forecasts = analysis["forecasts"]

    assert len(forecasts) == 3_650
    assert set(forecasts["year"]) == {2017}
    assert np.isfinite(forecasts["baseline_forecast"]).all()
    assert np.isfinite(forecasts["final_forecast"]).all()
    assert (forecasts["baseline_forecast"] >= 0).all()
    assert (forecasts["final_forecast"] >= 0).all()
    assert (forecasts["interval_upper"] >= forecasts["interval_lower"]).all()

    selected = analysis["validation_comparison"].query("selected")
    assert len(selected) == 1
    assert selected.iloc[0]["model"] == FINAL_MODEL_NAME


def test_final_model_beats_baseline_on_2016_and_2017() -> None:
    data = load_and_validate_data(DATA_PATH)
    analysis = run_full_analysis(data)
    validation = analysis["validation_comparison"].set_index("model")
    test = analysis["test_comparison"].set_index("model")

    assert validation.loc[FINAL_MODEL_NAME, "rmse"] < validation.loc[
        BASELINE_MODEL_NAME, "rmse"
    ]
    assert validation.loc[FINAL_MODEL_NAME, "mae"] < validation.loc[
        BASELINE_MODEL_NAME, "mae"
    ]
    assert test.loc[FINAL_MODEL_NAME, "rmse"] < test.loc[BASELINE_MODEL_NAME, "rmse"]
    assert test.loc[FINAL_MODEL_NAME, "mae"] < test.loc[BASELINE_MODEL_NAME, "mae"]


def test_final_model_follows_the_documented_practical_tie_rule() -> None:
    data = load_and_validate_data(DATA_PATH)
    analysis = run_full_analysis(data)
    validation = analysis["validation_comparison"]
    selected = validation.loc[validation["selected"]].iloc[0]
    best_rmse = validation["rmse"].min()
    practically_tied = validation[
        validation["rmse"] <= best_rmse * 1.001
    ]

    assert selected["rmse"] <= best_rmse * 1.001
    assert selected["parameters"] == practically_tied["parameters"].min()


def test_final_model_beats_baseline_in_every_store_month_cell() -> None:
    data = load_and_validate_data(DATA_PATH)
    forecasts = run_full_analysis(data)["forecasts"]

    for _, cell in forecasts.groupby(["store_id", "month"]):
        baseline = regression_metrics(cell["sales"], cell["baseline_forecast"])
        final = regression_metrics(cell["sales"], cell["final_forecast"])
        assert final["rmse"] < baseline["rmse"]

