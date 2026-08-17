# Naphtha Splitter Data-Driven Modelling Report

## Source basis
- DCS sheet: `DCS data shift wise`
- Laboratory sheet: `lab results swift wise`
- Modelling logic: supplied `naphtha_splitter_modelling_spec.md`
- Dashboard architecture: supplied `Naphtha Splitter Dashboard Web Preview Development Guide.md`

## Dataset alignment
The DCS workbook contains E-shift and N-shift operating averages. The laboratory sheet contains M/E/N shift laboratory records. For supervised modelling, only E and N records were used because those are the shifts for which matching DCS operating data are present.

Matched modelling records: **713**  
Date range: **2025-08-14 to 2026-08-13**

## Important target naming limitation
The supplied modelling specification describes the bottom product as **C90-160 / C90-160+**, while the supplied laboratory workbook contains a sample named **C5 90-120**. Therefore the model does **not** relabel 90-120 as C90-160. It models the laboratory targets actually present:
- C5-90 IBP / FBP
- C5 90-120 IBP / FBP

A future model for C90-160 should use laboratory data for that actual product.

## Model
Algorithm: ExtraTreesRegressor, 500 trees, minimum leaf size 5, max_features 0.9.

Inputs selected directly from the supplied specification and DCS data:
- NSU Reflux Flow
- NSU Top Temperature
- NSU Bottom Temperature
- NSU Top Pressure
- Side Cut Flow
- CDU Top Temperature
- Stabilizer Bottom Temperature

Validation: chronological holdout; first 80% train, last 20% test.

## Time-ordered test performance

| Target | Train | Test | MAE °C | R² |
|---|---:|---:|---:|---:|
| IBP_C5-90 | 570 | 143 | 1.608 | -0.567 |
| FBP_C5-90 | 570 | 143 | 2.258 | 0.069 |
| IBP_C5 90-120 | 570 | 143 | 2.269 | 0.220 |
| FBP_C5 90-120 | 570 | 143 | 3.877 | 0.274 |

## Interpretation
The current dataset supports a useful **prototype / what-if dashboard**, but the time-ordered validation results are not strong enough to claim production-grade assay prediction. In particular, negative or low R² means the model does not yet explain enough of the unseen temporal variation.

The dashboard therefore presents predictions as model-assisted estimates and explicitly warns against extrapolation.

## Recommended next modelling stage
1. Add actual C90-160/C90-160+ laboratory results if that is the intended controlled product.
2. Confirm DCS-to-lab sample collection time alignment and laboratory sampling delay.
3. Add additional feed/composition variables if available.
4. Add lagged process variables (for example 1–24 h) to account for column residence time and lab sampling delay.
5. Add data-quality filtering for unit upsets, analyzer/sample anomalies and operating transitions.
6. Compare ExtraTrees with XGBoost/LightGBM and regularized linear baselines.
7. Use walk-forward validation rather than a single holdout before APC deployment.
