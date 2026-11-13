# Bakery Forecast Lab

An interactive Streamlit app that compares management's promotion-only sales
model with a stronger daily forecasting model for one croissant item across ten
stores. The app uses 2017 only as an untouched out-of-sample test year.

## What the app delivers

- Select any store and one month or a continuous range of months in 2017
- Compare actual daily sales with the baseline and final forecasts
- Recalculate RMSE and MAE for the active store and period
- Inspect an empirical 80% forecast band calibrated with 2016 errors
- Review 2016 model selection, 2017 holdout performance, driver effects, and
  required residual diagnostics
- Read the assumptions, limitations, and modeling chronology without opening code
- Download the filtered daily forecast data

## Leakage-safe analysis plan

| Period | Purpose |
|---|---|
| 2013–2015 | Fit candidate models |
| 2016 | Compare candidates with RMSE and MAE; lock the final specification |
| 2013–2016 | Refit the locked baseline and final models |
| 2017 | Evaluate once and power the interactive dashboard |

The management baseline is:

```text
sales ~ promotion
```

The selected final model is:

```text
log(1 + sales) ~ promotion + bad_weather + store + weekday
```

The log model is returned to the original sales scale with Duan's smearing
correction. Predictions are clipped at zero. The displayed 80% band adds the
10th and 90th percentiles of the final candidate's 2016 validation errors to
each 2017 point forecast.

Model selection uses an explicit parsimony rule: candidates within 0.1% of the
lowest 2016 RMSE are treated as practically tied, and the model with the fewest
estimated parameters is selected. This keeps the decision reproducible and
prevents negligible validation differences from adding unnecessary terms.

## Project files

```text
bakery-forecasting-app/
├── app.py                    # Streamlit dashboard
├── modeling.py               # Data checks, models, splits, metrics, diagnostics
├── export_results.py         # Optional reproducible CSV/JSON exports
├── MODEL_REPORT.md            # Written statistical analysis and findings
├── ASSIGNMENT_AUDIT.md        # Requirement-by-requirement final QA review
├── SUBMISSION_CHECKLIST.md    # GitHub, deployment, and submission steps
├── requirements.txt          # Streamlit Cloud dependencies
├── runtime.txt               # Python version for deployment
├── data/
│   └── bakerysales.csv       # Supplied case data
├── tests/
│   └── test_modeling.py      # Data, leakage, metric, and forecast tests
└── .streamlit/
    └── config.toml           # App theme and server settings
```

## Run locally

1. Open a terminal in this project folder.
2. Create and activate a virtual environment if desired.
3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the tests:

   ```bash
   python -m pytest -q
   ```

5. Launch the app:

   ```bash
   streamlit run app.py
   ```

6. Open the local address Streamlit prints, usually
   `http://localhost:8501`.

## Publish through GitHub and Streamlit Community Cloud

1. Create a new public GitHub repository, for example
   `bakery-forecast-lab`.
2. Upload **the contents of this folder** to the repository root. `app.py`,
   `modeling.py`, and `requirements.txt` should be visible immediately when the
   repository opens. Keep `bakerysales.csv` inside the `data` folder.
3. Commit the files to the `main` branch.
4. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
5. Choose **Create app**, select the repository and `main` branch, and set the
   main file path to `app.py`.
6. Deploy. Streamlit will install `requirements.txt`, fit the small models once,
   and provide a public URL for submission.

If deployment says the repository does not exist, confirm that Streamlit's
GitHub app has permission to see the repository and that the repository owner,
name, and branch match exactly.

## Reproduce the result tables

The dashboard calculates results directly from the source CSV. To also create
portable CSV and JSON result files, run:

```bash
python export_results.py
```

This creates an `outputs` folder locally. That folder is excluded from Git
because it can always be rebuilt from the source data and code.

## Main limitations

- Results cover one product and may not generalize to the rest of the assortment
- Promotion and weather coefficients are associations rather than causal effects
- Future promotion and weather inputs must be available to create a forecast
- Price, holidays, stockouts, local events, and competitor activity are absent
- The empirical interval is a global planning range, not a formal store-specific
  conditional prediction interval
- Forecast accuracy does not directly optimize waste, shortages, or profit

