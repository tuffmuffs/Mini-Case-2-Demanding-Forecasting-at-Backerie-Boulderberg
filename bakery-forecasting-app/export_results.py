"""Export the reproducible model results used by the Streamlit app.

Run from the project root:
    python export_results.py

The app fits from the source CSV at startup, so these exports are supporting
evidence rather than runtime dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

from modeling import load_and_validate_data, run_full_analysis

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    """Run the full chronological analysis and write readable result files."""

    data = load_and_validate_data(ROOT / "data" / "bakerysales.csv")
    analysis = run_full_analysis(data)
    OUTPUT_DIR.mkdir(exist_ok=True)

    analysis["validation_comparison"].to_csv(
        OUTPUT_DIR / "2016_model_validation.csv", index=False
    )
    analysis["test_comparison"].to_csv(
        OUTPUT_DIR / "2017_holdout_metrics.csv", index=False
    )
    analysis["driver_changes"].to_csv(
        OUTPUT_DIR / "modeled_driver_changes.csv", index=False
    )
    analysis["forecasts"].to_csv(
        OUTPUT_DIR / "2017_daily_forecasts.csv", index=False
    )
    with (OUTPUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(analysis["summary"], file, indent=2)

    print(f"Wrote analysis outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

