# Assignment Compliance Audit

This checklist maps the finished project directly to every requirement in the
Mini-Project 2 instructions and records the main analytical pitfalls reviewed
before submission.

## Part 1: Model Development

| Assignment requirement | Where it is satisfied | Audit result |
|---|---|---|
| Use the 2013–2016 observations for model development | `modeling.py` uses 2013–2015 for candidate fitting, 2016 for internal validation, and refits the locked models on all 2013–2016 rows | Pass |
| Estimate management's promotion-only model | `BASELINE_FORMULA = "sales ~ promotion"` | Pass |
| Incorporate store, calendar, weather, and promotion effects | Alternatives include store fixed effects, weekday, month, bad weather, promotion, a quadratic promotion term, and a promotion-by-store-type interaction | Pass |
| Consider transformations and functional forms | Candidate set compares sales-level OLS with `log(1 + sales)`, linear and quadratic promotion, month effects, and an interaction | Pass |
| Select and explain a final model | 2016 RMSE and MAE are shown in the app and report; the documented 0.1% practical-tie rule selects the fewest-parameter near-best model | Pass |

### Why the internal split is acceptable

The assignment allows 2013–2016 for development and prohibits using 2017 to
estimate or select the model. Using 2016 as an internal validation year is a
stricter use of the allowed development period. After selection, the baseline
and final specifications are refitted on all 2013–2016 observations before 2017
predictions are produced.

## Part 2: Diagnostics and Forecast Evaluation

| Assignment requirement | Where it is satisfied | Audit result |
|---|---|---|
| Residual histogram | **Model evidence** tab, evaluated on the modeled log scale | Pass |
| Fitted-versus-residual plot | **Model evidence** tab, with a zero reference line | Pass |
| Discuss assumptions and limitations | Assumption-check table in the app plus detailed discussion in `MODEL_REPORT.md` | Pass |
| Keep 2017 out of estimation and selection | `prepare_splits()` isolates 2017; automated tests verify no overlap | Pass |
| Generate 2017 predictions for both models | `run_full_analysis()` refits both locked specifications and predicts all 3,650 store-days in 2017 | Pass |
| Compare RMSE and MAE | Overall 2017 table and filter-responsive app metrics | Pass |

## Required Interactive Deliverable

| Deliverable requirement | Implementation | Audit result |
|---|---|---|
| Show current and recommended models | Promotion-only baseline and final operational log forecast appear together | Pass |
| Select one of ten stores | Store selector includes stores 1–10 and store type | Pass |
| Select one month or continuous range in 2017 | Two-handle month-range selector allows identical endpoints or a continuous range | Pass |
| Daily actual and both forecasts | Interactive Plotly daily time series | Pass |
| RMSE and MAE for every displayed model and active period | Four metric cards recalculate after every filter change | Pass |
| Discuss limitations | Dedicated **Method & limitations** tab | Pass |
| Understandable without code or raw output | Plain-language labels, model-selection explanation, metric definitions, chronology, and interpretation are included in the app | Pass |

## Analytical Pitfalls Reviewed

### Data leakage

- No 2017 sales outcomes enter candidate fitting, selection, smearing correction,
  or uncertainty-band calibration
- The app calculates 2017 accuracy only after predictions are generated
- The 2017 weather and promotion fields are used as predictor inputs; this is
  permitted but means a real forecast requires those future inputs to be known

### Inconsistent model selection

The earlier draft retained a 19-parameter quadratic model even though its gain
over the 18-parameter log model was as small as the gain used to reject a
21-parameter interaction. The final version fixes that inconsistency with one
explicit rule: models within 0.1% of the lowest 2016 RMSE are practically tied,
and the model with fewer parameters wins. The 18-parameter operational log model
is selected without consulting 2017.

### Retransformation bias

Exponentiating a log prediction directly would systematically understate mean
sales. The project applies Duan's smearing correction before subtracting one and
returning predictions to the original unit scale.

### Invalid metric scale

RMSE and MAE are calculated on original daily sales units, not on logged sales.
This makes both measures operationally interpretable and directly comparable
between the baseline and final models.

### Store type and multicollinearity

Store type is constant within each store. Its main effect is not combined with
the full set of store fixed effects because those terms would be redundant. A
promotion-by-store-type interaction is considered as a candidate because the
promotion slope can still vary within stores over time.

### Count outcome and negative forecasts

Sales are counts, but the values are sufficiently large for the log-OLS approach
to be practical. The log response improves residual shape, and returned forecasts
are clipped at zero so the model cannot display negative sales.

### Dependence and causal interpretation

- Store-day observations form a panel, so complete independence cannot be guaranteed
- Mean within-store lag-one residual correlation is approximately 0.021, which is small
- Promotion and weather coefficients remain observational associations, not causal effects
- Store fixed effects support these ten stores but not an unseen eleventh store

### Uncertainty band

The optional 80% band uses 2016 validation-error quantiles and is fixed before
2017 evaluation. Overall 2017 coverage is about 79.1%, but coverage differs by
store, so the app labels it an empirical planning band rather than a formal
conditional prediction interval.

## Final Numerical Cross-Checks

| Measure | Baseline | Final model | Reduction |
|---|---:|---:|---:|
| 2017 RMSE | 79.32 | 44.32 | 44.1% |
| 2017 MAE | 61.22 | 32.60 | 46.8% |

- The final model beats the baseline in all 120 store-by-month cells in 2017
- Log-residual skewness is approximately -0.001
- Correlation between absolute log residuals and fitted log sales is approximately -0.001
- Eight automated tests pass in a fresh Python 3.12 environment
- Static code checks report no undefined names or syntax-level errors

## Remaining Limitations to State Honestly

- One product and ten existing stores limit generalizability
- Only one year is used for internal validation
- Holidays, price, stockouts, local events, and competitor activity are absent
- Future promotion and weather values must be available
- The app optimizes forecast accuracy, not the asymmetric costs of waste and stockouts

