import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="NSU Predictive Dashboard", layout="wide")

XLSX = "Stabilizer & NSU data sets(1).xlsx"

@st.cache_data
def load_data(path):
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
      "TOP T":"cdu_top_temp"}
    return d.rename(columns=rename)

FEATURES = ["reflux_flow","top_temp","bottom_temp","pressure","side_draw","cdu_top_temp","stab_bottom_t"]
TARGETS = ["IBP_C5-90","FBP_C5-90","IBP_C5 90-120","FBP_C5 90-120"]

@st.cache_resource
def train_models(df):
    d = df[["date","shift"]+FEATURES+TARGETS].copy()
    for c in FEATURES+TARGETS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna().sort_values(["date","shift"]).reset_index(drop=True)
    cut = int(len(d)*0.80)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    models, rows, predictions = {}, [], te[["date","shift"]].copy()
    for y in TARGETS:
        model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5,
                                    max_features=0.9, random_state=42, n_jobs=-1)
        model.fit(tr[FEATURES], tr[y])
        p = model.predict(te[FEATURES])
        models[y] = model
        predictions[y+"_actual"] = te[y].values
        predictions[y+"_pred"] = p
        rows.append([y, len(tr), len(te), mean_absolute_error(te[y],p), r2_score(te[y],p)])
    metrics = pd.DataFrame(rows, columns=["Target","Train","Test","MAE","R2"])
    return d, tr, te, models, metrics, predictions

df = load_data(XLSX)
d, tr, te, models, metrics, predictions = train_models(df)

st.title("Naphtha Splitter Unit — Predictive Process Dashboard")
st.caption("Data-driven model trained from the supplied DCS + laboratory dataset. This is a modelling prototype, not a validated APC/DCS controller.")

# Sidebar controls
st.sidebar.header("NSU Operating Inputs")
def slider(label, col, step):
    lo, hi = float(d[col].quantile(0.01)), float(d[col].quantile(0.99))
    default = float(d[col].median())
    return st.sidebar.slider(label, lo, hi, default, step=step)

reflux = slider("NSU Reflux Flow (T/hr)", "reflux_flow", 0.1)
top_t = slider("NSU Top Temperature (°C)", "top_temp", 0.1)
bottom_t = slider("NSU Bottom Temperature (°C)", "bottom_temp", 0.1)
pressure = slider("NSU Top Pressure (kg/cm²g)", "pressure", 0.01)
side = slider("Side Cut Flow (T/hr)", "side_draw", 0.1)
cdu_t = slider("CDU Top Temperature (°C)", "cdu_top_temp", 0.1)
stab_t = slider("Stabilizer Bottom Temperature (°C)", "stab_bottom_t", 0.1)

x = pd.DataFrame([{
    "reflux_flow":reflux, "top_temp":top_t, "bottom_temp":bottom_t,
    "pressure":pressure, "side_draw":side, "cdu_top_temp":cdu_t,
    "stab_bottom_t":stab_t
}])

pred = {y: float(models[y].predict(x)[0]) for y in TARGETS}

c1,c2,c3,c4 = st.columns(4)
c1.metric("C5-90 IBP", f"{pred['IBP_C5-90']:.1f} °C")
c2.metric("C5-90 FBP", f"{pred['FBP_C5-90']:.1f} °C")
c3.metric("90-120 IBP", f"{pred['IBP_C5 90-120']:.1f} °C")
c4.metric("90-120 FBP", f"{pred['FBP_C5 90-120']:.1f} °C")

tab1,tab2,tab3,tab4 = st.tabs(["Live Prediction","Historical Trends","Model Validation","Process Logic"])

with tab1:
    st.subheader("Current model prediction")
    st.dataframe(pd.DataFrame({
        "Variable":["Reflux Flow","Top T","Bottom T","Pressure","Side Cut","CDU Top T","Stabilizer Bottom T"],
        "Value":[reflux,top_t,bottom_t,pressure,side,cdu_t,stab_t],
        "Unit":["T/hr","°C","°C","kg/cm²g","T/hr","°C","°C"]
    }), use_container_width=True)
    st.info("Use the sliders as a scenario/what-if interface. Predictions are interpolations within the training-data envelope; avoid extrapolation beyond the observed ranges.")

with tab2:
    st.subheader("Observed operating history")
    cols = ["date","shift","top_temp","bottom_temp","pressure","reflux_flow","side_draw","cdu_top_temp","stab_bottom_t",
            "IBP_C5-90","FBP_C5-90","IBP_C5 90-120","FBP_C5 90-120"]
    st.line_chart(d.set_index("date")[["top_temp","bottom_temp"]])
    st.dataframe(d[cols].sort_values("date", ascending=False).head(100), use_container_width=True)

with tab3:
    st.subheader("Time-ordered holdout validation")
    st.dataframe(metrics.style.format({"MAE":"{:.2f}","R2":"{:.2f}"}), use_container_width=True)
    st.caption("Validation uses the last 20% of matched E/N shift records as an unseen time-ordered test set.")
    target = st.selectbox("Target", TARGETS)
    pcol = target+"_pred"; acol = target+"_actual"
    chart = predictions[["date",pcol,acol]].set_index("date").rename(columns={pcol:"Predicted",acol:"Actual"})
    st.line_chart(chart)
    fi = pd.Series(models[target].feature_importances_, index=FEATURES).sort_values(ascending=False)
    st.bar_chart(fi)

with tab4:
    st.subheader("Specified process cause-and-effect")
    st.markdown("""
- Higher reflux → lower top-product FBP and higher 90+ cut IBP.
- Higher NSU top temperature → higher top-product FBP and higher heavy-cut IBP.
- Higher bottom temperature / reboiler duty → higher heavy-cut IBP, but can raise top FBP if uncompensated.
- Higher pressure → pressure/VLE-driven shifts in predicted cut points.
- Higher side-cut draw → lower top FBP and higher heavy-cut IBP.
- Higher CDU top temperature → heavier feed envelope.
- Lower stabilizer bottom temperature → LPG slippage and lower top-product IBP.

These relationships are taken from the supplied modelling specification. They should be treated as engineering constraints/interpretation, not proof that the historical dataset contains all causal effects.
""")

st.sidebar.markdown("---")
st.sidebar.write(f"Matched modelling records: **{len(d)}**")
st.sidebar.write(f"Date range: **{d.date.min().date()} → {d.date.max().date()}**")
st.sidebar.write("Lab targets available: **C5-90** and **C5 90-120**.")
