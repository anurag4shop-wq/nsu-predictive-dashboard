import sys
sys.path.append(r'D:\Users\Asus\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages')

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="NSU Predictive Dashboard", layout="wide")

XLSX = "Stabilizer & NSU data sets(1).xlsx"

@st.cache_data
def load_data(path):
    # 1. DCS sheet
    raw = pd.read_excel(path, sheet_name="DCS data shift wise", header=None)
    dates = pd.to_datetime(raw.iloc[4, 5:], errors="coerce")
    blocks = []
    for shift, start, end in [("E",5,28),("N",31,54)]:
        rows = {}
        for i in range(start, end):
            name = raw.iloc[i,3]
            if pd.notna(name):
                rows[str(name).strip()] = pd.to_numeric(raw.iloc[i,5:], errors="coerce").values
        df = pd.DataFrame(rows, index=dates).reset_index()
        df = df.rename(columns={df.columns[0]:"date"})
        df["shift"] = shift
        blocks.append(df)
    dcs = pd.concat(blocks, ignore_index=True)

    # 2. Lab sheet
    lab = pd.read_excel(path, sheet_name="lab results swift wise", header=None)
    ld = lab.iloc[5:].copy()
    ld["sample"] = ld.iloc[:,1].ffill()
    ld["date"] = pd.to_datetime(ld.iloc[:,2].ffill(), format="%d.%m.%Y", errors="coerce")
    ld["shift"] = ld.iloc[:,3]
    lab2 = ld[ld["shift"].isin(["M","E","N"]) & ld["sample"].isin(["C5-90","C5 90-120"])].copy()
    lab2["IBP"] = pd.to_numeric(lab2.iloc[:,4], errors="coerce")
    lab2["FBP"] = pd.to_numeric(lab2.iloc[:,10], errors="coerce")
    piv = lab2.pivot_table(index=["date","shift"], columns="sample",
                           values=["IBP","FBP"], aggfunc="first").reset_index()
    piv.columns = ["_".join([str(x) for x in c if str(x)!=""]) if isinstance(c,tuple) else c
                   for c in piv.columns]
    d = dcs.merge(piv, on=["date","shift"], how="inner")
    
    rename = {
      "Stabilizer Feed Flow":"stab_feed","Stabilizer Top T":"stab_top_t",
      "Stabilizer Top P":"stab_top_p","Stabilizer Reflux drum T":"stab_reflux_drum_t",
      "Stabilizer Feed T":"stab_feed_t","Stabilizer Bottom T":"stab_bottom_t",
      "Stab Off Gas flow":"stab_offgas","LPG Flow":"lpg_flow",
      "Stabilized Naphtha Flow":"stab_naphtha_flow","Stabilizer Reflux Flow":"stab_reflux_flow",
      "Stabilizer Reboiler Duty-LGO CR":"stab_reboiler","NSU Feed Flow":"nsu_feed",
      "NSU Top T":"top_temp","NSU Top P":"pressure","NSU Reflux Drum T":"reflux_drum_t",
      "NSU Feed T":"feed_temp","NSU Bottom T":"bottom_temp","C5-90 Flow":"c5_90_flow",
      "Side cut Flow":"side_draw","90-120 Flow":"flow_90_120",
      "NSU Reflux Flow":"reflux_flow","Reboiler MP steam flow":"reboiler_steam",
      "TOP T":"cdu_top_temp"
    }
    df_merged = d.rename(columns=rename)

    # Automated Anomaly & Data Quality Filtering
    # 1. Operational shutdown / zero flow filtering
    valid_ops = (
        (df_merged["nsu_feed"] > 5) & 
        (df_merged["reflux_flow"] > 1) & 
        (df_merged["reboiler_steam"] > 0.5) &
        (df_merged["top_temp"] > 30) &
        (df_merged["bottom_temp"] > 50) &
        (df_merged["pressure"] > 0.1)
    )
    df_clean = df_merged[valid_ops].copy()
    
    # 2. 3-Sigma Lab Target Outlier Rejection
    targets = ["IBP_C5-90","FBP_C5-90","IBP_C5 90-120","FBP_C5 90-120"]
    for y in targets:
        mean = df_clean[y].mean()
        std = df_clean[y].std()
        df_clean = df_clean[(df_clean[y] >= mean - 3*std) & (df_clean[y] <= mean + 3*std)]
        
    return df_clean.sort_values(["date","shift"]).reset_index(drop=True)

FEATURES = [
    "reflux_flow", "top_temp", "bottom_temp", "pressure", "side_draw",
    "cdu_top_temp", "stab_bottom_t", "nsu_feed", "feed_temp", "reboiler_steam"
]
TARGETS = ["IBP_C5-90", "FBP_C5-90", "IBP_C5 90-120", "FBP_C5 90-120"]

class HybridPredictor:
    """
    Hybrid Regularized Model: Combines Scaled Ridge Linear Regression for concept drift
    & trend extrapolation with ExtraTreesRegressor for non-linear process dynamics.
    """
    def __init__(self):
        self.ridge_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-2, 3, 20)))
        ])
        self.tree = ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.9, random_state=42, n_jobs=-1
        )
        self.w_ridge = 0.5
        self.w_tree = 0.5

    def fit(self, X, y):
        self.ridge_pipe.fit(X, y)
        self.tree.fit(X, y)
        return self

    def predict(self, X):
        p_ridge = self.ridge_pipe.predict(X)
        p_tree = self.tree.predict(X)
        return self.w_ridge * p_ridge + self.w_tree * p_tree

    def feature_importances(self, feature_names):
        # Scale-adjusted Linear Coefficients magnitude + Tree Feature Importances
        ridge_coefs = np.abs(self.ridge_pipe.named_steps['ridge'].coef_)
        if ridge_coefs.sum() > 0:
            ridge_norm = ridge_coefs / ridge_coefs.sum()
        else:
            ridge_norm = np.zeros_like(ridge_coefs)
        tree_imp = self.tree.feature_importances_
        combined = 0.5 * ridge_norm + 0.5 * tree_imp
        return pd.Series(combined, index=feature_names).sort_values(ascending=False)

@st.cache_resource
def train_models(df):
    d = df[["date","shift"] + FEATURES + TARGETS].copy()
    for c in FEATURES + TARGETS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values(["date","shift"]).reset_index(drop=True)
    
    cut = int(len(d) * 0.80)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    
    models, rows, predictions = {}, [], te[["date","shift"]].copy()
    
    # 5-Fold Walk-Forward Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    for y in TARGETS:
        # 1. Evaluate 5-Fold Walk-Forward CV
        cv_r2_list, cv_mae_list = [], []
        for train_idx, test_idx in tscv.split(d):
            cv_tr, cv_te = d.iloc[train_idx], d.iloc[test_idx]
            cv_m = HybridPredictor()
            cv_m.fit(cv_tr[FEATURES], cv_tr[y])
            cv_p = cv_m.predict(cv_te[FEATURES])
            cv_r2_list.append(r2_score(cv_te[y], cv_p))
            cv_mae_list.append(mean_absolute_error(cv_te[y], cv_p))
            
        cv_r2 = np.mean(cv_r2_list)
        cv_mae = np.mean(cv_mae_list)
        
        # 2. Train on 80/20 Holdout
        model = HybridPredictor()
        model.fit(tr[FEATURES], tr[y])
        p = model.predict(te[FEATURES])
        
        models[y] = model
        predictions[y + "_actual"] = te[y].values
        predictions[y + "_pred"] = p
        
        holdout_mae = mean_absolute_error(te[y], p)
        holdout_r2 = r2_score(te[y], p)
        holdout_rmse = root_mean_squared_error(te[y], p)
        
        rows.append([y, len(tr), len(te), holdout_mae, holdout_r2, holdout_rmse, cv_mae, cv_r2])
        
    metrics = pd.DataFrame(rows, columns=[
        "Target", "Train Rows", "Test Rows", 
        "Holdout MAE (°C)", "Holdout R²", "Holdout RMSE (°C)",
        "5-Fold CV MAE (°C)", "5-Fold CV R²"
    ])
    return d, tr, te, models, metrics, predictions

df = load_data(XLSX)
d, tr, te, models, metrics, predictions = train_models(df)

st.title("Naphtha Splitter Unit — Advanced Predictive Process Dashboard")
st.caption("Data-driven Hybrid Regularized Model (Ridge + ExtraTrees) trained on DCS + laboratory dataset. Incorporates automated anomaly filtering and process gain matrix alignment.")

# Sidebar controls
st.sidebar.header("NSU Operating Inputs (10 Variables)")
def slider(label, col, step):
    lo, hi = float(d[col].quantile(0.01)), float(d[col].quantile(0.99))
    default = float(d[col].median())
    return st.sidebar.slider(label, lo, hi, default, step=step)

reflux = slider("1. NSU Reflux Flow (T/hr)", "reflux_flow", 0.1)
top_t = slider("2. NSU Top Temperature (°C)", "top_temp", 0.1)
bottom_t = slider("3. NSU Bottom Temperature (°C)", "bottom_temp", 0.1)
pressure = slider("4. NSU Top Pressure (kg/cm²g)", "pressure", 0.01)
side = slider("5. Side Cut Flow (T/hr)", "side_draw", 0.1)
cdu_t = slider("6. CDU Top Temperature (°C)", "cdu_top_temp", 0.1)
stab_t = slider("7. Stabilizer Bottom Temperature (°C)", "stab_bottom_t", 0.1)
nsu_feed = slider("8. NSU Feed Flow (T/hr)", "nsu_feed", 0.1)
feed_t = slider("9. NSU Feed Temperature (°C)", "feed_temp", 0.1)
reboiler_steam = slider("10. Reboiler MP Steam Flow (T/hr)", "reboiler_steam", 0.1)

x = pd.DataFrame([{
    "reflux_flow": reflux, "top_temp": top_t, "bottom_temp": bottom_t,
    "pressure": pressure, "side_draw": side, "cdu_top_temp": cdu_t,
    "stab_bottom_t": stab_t, "nsu_feed": nsu_feed, "feed_temp": feed_t,
    "reboiler_steam": reboiler_steam
}])

pred = {y: float(models[y].predict(x)[0]) for y in TARGETS}

c1, c2, c3, c4 = st.columns(4)
c1.metric("C5-90 IBP (Overhead Initial)", f"{pred['IBP_C5-90']:.1f} °C")
c2.metric("C5-90 FBP (Overhead Final)", f"{pred['FBP_C5-90']:.1f} °C")
c3.metric("90-160 IBP (Bottoms Initial)", f"{pred['IBP_C5 90-120']:.1f} °C")
c4.metric("90-160 FBP (Bottoms Final)", f"{pred['FBP_C5 90-120']:.1f} °C")

tab1, tab2, tab3, tab4 = st.tabs(["Live Scenario Simulation", "Historical Trends", "Model Validation & Metrics", "Process Control Gain Matrix"])

with tab1:
    st.subheader("Current Operating Scenario & Input State")
    st.dataframe(pd.DataFrame({
        "Process Variable": [
            "NSU Reflux Flow", "NSU Top Temperature", "NSU Bottom Temperature", 
            "NSU Top Pressure", "Side Cut Flow", "CDU Top Temperature", 
            "Stabilizer Bottom Temperature", "NSU Feed Flow", "NSU Feed Temperature", "Reboiler MP Steam Flow"
        ],
        "Input Value": [reflux, top_t, bottom_t, pressure, side, cdu_t, stab_t, nsu_feed, feed_t, reboiler_steam],
        "Unit": ["T/hr", "°C", "°C", "kg/cm²g", "T/hr", "°C", "°C", "T/hr", "°C", "T/hr"],
        "Gain Matrix Category": ["MV", "MV", "MV", "MV", "MV", "DV (Feedforward)", "DV", "MV", "DV", "MV"]
    }), use_container_width=True)
    st.info("The scenario simulator uses a hybrid regularized ensemble (Ridge + ExtraTrees) to maintain trend stability outside historical boundaries. Note: C5 90-120 and 90-160 cuts are treated as equivalent in plant operational specs.")

with tab2:
    st.subheader("Filtered Operational History")
    cols = ["date","shift","nsu_feed","reflux_flow","top_temp","bottom_temp","pressure","side_draw","reboiler_steam",
            "IBP_C5-90","FBP_C5-90","IBP_C5 90-120","FBP_C5 90-120"]
    st.line_chart(d.set_index("date")[["top_temp","bottom_temp","feed_temp"]])
    st.dataframe(d[cols].sort_values("date", ascending=False).head(100), use_container_width=True)

with tab3:
    st.subheader("Model Performance & Validation Metrics")
    st.dataframe(metrics.style.format({
        "Holdout MAE (°C)": "{:.2f}", "Holdout R²": "{:.3f}", "Holdout RMSE (°C)": "{:.2f}",
        "5-Fold CV MAE (°C)": "{:.2f}", "5-Fold CV R²": "{:.3f}"
    }), use_container_width=True)
    st.caption("Validation metrics compare an 80/20 chronological holdout test set with 5-Fold Walk-Forward Time-Series Cross Validation.")
    
    target = st.selectbox("Select Target Cut Point", TARGETS)
    pcol = target + "_pred"
    acol = target + "_actual"
    chart = predictions[["date", pcol, acol]].set_index("date").rename(columns={pcol: "Model Predicted", acol: "Lab Actual"})
    st.line_chart(chart)
    
    st.subheader(f"Combined Feature Importance ({target})")
    fi = models[target].feature_importances(FEATURES)
    st.bar_chart(fi)

with tab4:
    st.subheader("Naphtha Splitter Unit (NSU) — Multivariable Process Control Gain Matrix")
    st.markdown("""
    **Multivariable Predictive Control - Relative Weightage (%) & Directional Gain Signs (+ / -)**
    """)
    
    gain_matrix_df = pd.DataFrame([
        {"Process Variable": "1. Stabilizer Bottom T", "C5-90 IBP": "+ (40%)", "C5-90 FBP": "0%", "90-160 IBP": "0%", "90-160 FBP": "0%"},
        {"Process Variable": "2. NSU Top P", "C5-90 IBP": "+ (25%)", "C5-90 FBP": "+ (15%)", "90-160 IBP": "+ (15%)", "90-160 FBP": "0%"},
        {"Process Variable": "3. NSU Reflux Flow", "C5-90 IBP": "+ (10%)", "C5-90 FBP": "- (45%)", "90-160 IBP": "- (25%)", "90-160 FBP": "0%"},
        {"Process Variable": "4. Side Cut Flow", "C5-90 IBP": "0%", "C5-90 FBP": "+ (10%)", "90-160 IBP": "+ (25%)", "90-160 FBP": "0%"},
        {"Process Variable": "5. Reboiler MP Steam Flow (and NSU Bottom T)", "C5-90 IBP": "- (25%)", "C5-90 FBP": "+ (30%)", "90-160 IBP": "+ (25%)", "90-160 FBP": "0%"},
        {"Process Variable": "6. CDU Column Top T", "C5-90 IBP": "0%", "C5-90 FBP": "0%", "90-160 IBP": "0%", "90-160 FBP": "+ (100%)"}
    ])
    st.table(gain_matrix_df)
    
    st.markdown("""
    #### Operational Guidelines & Gain Signs Legend:
    * **`(+)` Sign**: Increasing the Process Handle increases the Controlled Quality / Cut Point.
    * **`(-)` Sign**: Increasing the Process Handle decreases the Controlled Quality / Cut Point.
    * **CDU Top T**: Acts as a pure Feedforward Variable fixing the 90-160 FBP envelope (100% impact).
    * **Side Cut Flow**: Primary intermediate splitter between C5-90 FBP and 90-160 IBP with 0% impact on column boundaries.
    * **NSU Reflux Flow**: Primary handle for lowering top-product FBP (-45%) and bottoms IBP (-25%).
    * **Reboiler MP Steam Flow**: Primary energy handle raising top FBP (+30%) and bottoms IBP (+25%).
    """)

st.sidebar.markdown("---")
st.sidebar.write(f"Filtered Matched Records: **{len(d)}**")
st.sidebar.write(f"Date Range: **{d.date.min().date()} → {d.date.max().date()}**")
st.sidebar.write("Model Architecture: **Hybrid Ridge Linear + ExtraTrees Ensemble**")
