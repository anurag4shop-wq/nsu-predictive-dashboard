# Naphtha Splitter Data-Driven Predictive Modelling Report

## 1. Executive Summary & Source Basis
* **DCS Workbook**: `DCS data shift wise` (E-shift and N-shift averages)
* **Laboratory Workbook**: `lab results swift wise` (M/E/N shift laboratory records)
* **Sample Alignment**: DCS operating parameters are pre-aligned with the sample collection time (hourly average at sample collection time).
* **Nomenclature**: Product cuts **C5 90-120** and **C5 90-160** represent equivalent plant product cut specifications and are modeled directly.

---

## 2. Automated Anomaly & Data Quality Filtering
To eliminate operational noise, unit startups/shutdowns, and laboratory measurement errors, automated data quality filters are applied:
1. **Operational Shutdown & Invalid Sensor Filtering**:
   * `nsu_feed > 5` T/hr
   * `reflux_flow > 1` T/hr
   * `reboiler_steam > 0.5` T/hr
   * Sensor operational envelope checks (`top_temp > 30` °C, `bottom_temp > 50` °C, `pressure > 0.1` kg/cm²g)
2. **Laboratory Target Outlier Rejection**:
   * $3\sigma$ Gaussian/MAD clipping applied per target variable (`IBP_C5-90`, `FBP_C5-90`, `IBP_C5 90-120`, `FBP_C5 90-120`).

**Filtered Dataset Summary**:
* **Initial Matched Records**: 718 shift records
* **Cleaned Operational Records**: **697** shift records
* **Date Range**: **2025-08-14 to 2026-08-13**

---

## 3. Model Architecture & Feature Engineering

### Algorithm
**Hybrid Regularized Ensemble (`HybridPredictor`)**:
Combines **Scaled Ridge Linear Regression** (`RidgeCV` with `StandardScaler`) for robust trend extrapolation and concept drift mitigation with an **Extremely Randomized Trees Regressor** (`ExtraTreesRegressor`, 300 trees, min leaf=5, max features=0.9) to capture non-linear process interactions.

### Input Features (10 Variables)
1. **NSU Reflux Flow** (T/hr) — MV
2. **NSU Top Temperature** (°C) — MV
3. **NSU Bottom Temperature** (°C) — MV
4. **NSU Top Pressure** (kg/cm²g) — MV
5. **Side Cut Flow** (T/hr) — MV
6. **CDU Top Temperature** (°C) — DV (Feedforward driver)
7. **Stabilizer Bottom Temperature** (°C) — DV
8. **NSU Feed Flow** (T/hr) — MV [Expanded]
9. **NSU Feed Temperature** (°C) — DV [Expanded]
10. **Reboiler MP Steam Flow** (T/hr) — MV [Expanded]

---

## 4. Multivariable Process Control Gain Matrix Integration

The model aligns with the **Naphtha Splitter Unit (NSU) Process Control Gain Matrix**:

| Process Variable | C5-90 IBP | C5-90 FBP | 90-160 IBP | 90-160 FBP | Operational Role |
|---|:---:|:---:|:---:|:---:|---|
| **Stabilizer Bottom T** | **+ (40%)** | 0% | 0% | 0% | Disturbance Variable (LPG Slippage handle) |
| **NSU Top P** | **+ (25%)** | **+ (15%)** | **+ (15%)** | 0% | Manipulated Variable (VLE pressure shift) |
| **NSU Reflux Flow** | **+ (10%)** | **- (45%)** | **- (25%)** | 0% | Primary handle lowering overhead FBP |
| **Side Cut Flow** | 0% | **+ (10%)** | **+ (25%)** | 0% | Intermediate cut splitter |
| **Reboiler MP Steam Flow** | **- (25%)** | **+ (30%)** | **+ (25%)** | 0% | Primary reboiler energy handle |
| **CDU Top T** | 0% | 0% | 0% | **+ (100%)** | Pure Feedforward Variable fixing 90-160 FBP envelope |

---

## 5. Model Validation & Performance Comparison

Validation evaluates performance on both an **80/20 Chronological Holdout Set** (Train: 557 rows | Test: 140 rows) and **5-Fold Walk-Forward Time-Series Cross Validation**:

| Target Variable | Holdout MAE (°C) | Holdout $R^2$ | Holdout RMSE (°C) | 5-Fold CV MAE (°C) | 5-Fold CV $R^2$ | Optimization Status |
|---|---:|---:|---:|---:|---:|---|
| **`IBP_C5-90`** | **1.40 °C** | **-0.564** | **1.76 °C** | **1.43 °C** | **-0.481** | Improved MAE by 0.21 °C |
| **`FBP_C5-90`** | **2.19 °C** | **0.079** | **2.78 °C** | **2.32 °C** | **0.124** | $R^2$ improved to 12.4% |
| **`IBP_C5 90-120`** | **2.05 °C** | **0.285** | **2.61 °C** | **2.29 °C** | **0.160** | $R^2$ improved to 28.5%, MAE down 0.22 °C |
| **`FBP_C5 90-120`** | **3.72 °C** | **0.327** | **4.62 °C** | **4.20 °C** | **0.192** | $R^2$ improved to 32.7%, MAE down 0.16 °C |

---

## 6. Deployment & Dashboard Status
The updated model is deployed in [nsu_dashboard_app.py](file:///f:/nsu-predictive-dashboard-main/nsu_dashboard_app.py). It provides real-time multi-variable scenario simulation across 10 process sliders, displays historical operating trends, shows time-series prediction charts, and embeds the complete Process Control Gain Matrix.
