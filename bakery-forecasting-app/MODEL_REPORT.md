# Bakery Forecast Lab: Statistical Analysis

## Executive result

The final model materially improves daily croissant forecasts over management's
promotion-only baseline. On the untouched 2017 test year, RMSE falls from
**79.32 to 44.32 units** (a **44.1% reduction**) and MAE falls from **61.22 to
32.60 units** (a **46.8% reduction**).

The result is credible because the model specification was locked before 2017
was evaluated. Candidate models were fitted on 2013–2015, compared on 2016, and
then the selected baseline and final models were refitted on 2013–2016. The 2017
data served only as the final out-of-sample test and interactive dashboard period.

## 1. Data and quality checks

The supplied data contain 18,260 store-day observations for one croissant item:

- 10 stores
- 1,826 consecutive days per store from January 1, 2013 through December 31, 2017
- No missing values in required fields
- No duplicated store-date combinations
- Daily sales, promotion, bad-weather indicator, weekday, month, store ID, and store type

Store type is constant within a store, so the final model uses store fixed effects
rather than including both store and store type as redundant main effects.

## 2. Management baseline

The required baseline is an ordinary least-squares model:

```text
sales ~ promotion
```

This model is intentionally simple. It can respond to promotion depth, but it
cannot distinguish a high-volume urban store from a lower-volume rural store,
weekday from weekend demand, or normal from bad weather.

## 3. Candidate model development

All candidate models were fitted using 2013–2015 and evaluated on 2016. Lower
RMSE and MAE indicate better forecasts.

| Candidate | 2016 RMSE | 2016 MAE | Parameters | Decision |
|---|---:|---:|---:|---|
| Log + promotion by store type | 43.55 | 32.15 | 21 | Not selected |
| Operational log + curved promotion | 43.56 | 32.15 | 19 | Not selected |
| Operational log | 43.57 | 32.17 | 18 | **Selected** |
| Log + curved promotion + month | 43.65 | 32.21 | 30 | Not selected |
| Operational additive on sales scale | 45.96 | 34.13 | 18 | Not selected |
| Promotion-only baseline | 78.55 | 60.72 | 2 | Required benchmark |

The model-selection rule treats candidates within 0.1% of the lowest validation
RMSE as practically tied and then chooses the fewest parameters. The three log
models with 18, 19, and 21 parameters all met that threshold. The 18-parameter
operational log model was only 0.026 RMSE units, or about 0.06%, above the
absolute minimum, while MAE differed by only 0.020 units. It was therefore
selected to avoid adding a quadratic or interaction for a negligible validation
gain. Adding month also failed to improve validation accuracy.

The selected specification is:

```text
log(1 + sales) ~ promotion + bad_weather + store + weekday
```

The log transformation addresses the right-skewed sales distribution and makes
residual spread more consistent. Store and weekday are categorical fixed effects.
Predictions are returned to the original sales scale with Duan's smearing
correction and clipped at zero.

## 4. Final out-of-sample test

After model selection, both locked specifications were refitted on all data from
2013–2016 and evaluated once on 2017.

| Model | 2017 RMSE | 2017 MAE | Mean bias |
|---|---:|---:|---:|
| Promotion-only baseline | 79.32 | 61.22 | -2.06 |
| Final operational model | **44.32** | **32.60** | -1.95 |

The final model improves both measures by a similar amount, so the conclusion is
not driven by a single error definition. Both models slightly underpredict on
average, but the mean bias is small relative to their typical daily errors.

## 5. Operational interpretation

Holding the reference store and other listed conditions constant, the final
model associates:

- A promotion increase from 0% to 10% with approximately **12.6% higher** daily sales
- Bad weather with approximately **20.0% lower** daily sales
- Saturday with approximately **56.7% higher** sales than Monday
- Sunday with approximately **55.9% higher** sales than Monday

These are model associations, not causal estimates. Promotions may be scheduled
in anticipation of demand, and weather may coincide with other unobserved factors.

## 6. Diagnostics and uncertainty

The required residual histogram and residual-versus-fitted plot are available in
the app's **Model evidence** tab. Residuals are assessed on the modeled
`log(1 + sales)` scale. Their skewness is approximately **0.00**, indicating that
the log transformation produced a nearly symmetric residual distribution.
Some large residuals remain on high-demand days, and observations on consecutive
days may not be fully independent.

The dashboard also displays an empirical 80% planning band. Its lower and upper
adjustments are the 10th and 90th percentiles of 2016 validation errors (-48.73
and +52.67 units). Applied without recalibration to 2017, the band covers **79.1%**
of observations, close to its intended 80% rate. This is useful as a transparent
uncertainty summary but is not a formal conditional prediction interval.

## 7. Regression assumptions

- **Functional form:** The log response and categorical store and weekday effects
  remove the strong structure visible in the simpler models. The residual plot
  has no obvious remaining curve, although omitted holiday or local-event effects
  could still matter.
- **Residual normality:** Log-residual skewness is -0.001, which is nearly
  symmetric. Individual high-demand days can still produce large errors.
- **Constant variance:** The correlation between absolute residual size and
  fitted log sales is -0.001, consistent with stable residual spread after the
  log transformation. The plot is still the primary diagnostic.
- **Independent errors:** Mean within-store lag-one residual correlation is
  0.021, which is small. Repeated daily observations can nevertheless retain
  dependence that ordinary OLS does not explicitly model.
- **Exogenous predictors:** This cannot be established from observational data.
  Promotion timing may respond to expected demand, so coefficients should not be
  read as causal effects.
- **Multicollinearity:** Store type is constant within a store, so its main effect
  is intentionally excluded when store fixed effects are present.

## 8. Limitations

- The analysis covers one croissant item and may not generalize to other products
- Future promotion and weather inputs must be known to generate these forecasts
- Store fixed effects assume each store's underlying demand level remains stable
- Store fixed effects cannot forecast a brand-new store that was absent from training
- Price, holidays, stockouts, local events, and competitor activity are unavailable
- OLS treats store-day errors as independent even though serial correlation may exist
- Model selection relies on one validation year rather than rolling-origin validation
- The global empirical band may be too wide or narrow for a specific store or weekday
- Forecast error does not directly measure the cost of waste or missed sales

The clearest business extension is to attach asymmetric costs to leftovers and
stockouts, then convert the demand forecast into a recommended bake quantity for
each store-day.

