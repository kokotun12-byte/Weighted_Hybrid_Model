import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from tensorflow.keras.models import load_model


st.set_page_config(page_title="Polymer Import Forecast", page_icon="📈", layout="wide")

ARTIFACT_DIR = "artifacts"

@st.cache_resource
def load_artifacts():
    metadata = joblib.load(os.path.join(ARTIFACT_DIR, "hybrid_metadata.pkl"))
    arimax_fit = joblib.load(os.path.join(ARTIFACT_DIR, "final_arimax_fit.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    history_df = joblib.load(os.path.join(ARTIFACT_DIR, "history_df.pkl"))
    lstm_model = tf.keras.models.load_model(
    "artifacts/lstm_model.h5",
    compile=False)
    return metadata, arimax_fit, scaler, history_df, lstm_model

metadata, arimax_fit, scaler, history_df, lstm_model = load_artifacts()

FEATURE_COLS = metadata["feature_cols"]
EXOG_LOG = metadata["exog_log"]
LOOKBACK = int(metadata["lookback"])
BEST_WEIGHT = float(metadata["best_weight"])
LAST_HISTORY_DATE = pd.to_datetime(metadata["last_date"])

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.metric-card {
    background: #ffffff;
    padding: 1.2rem;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #eef0f2;
}
.small-note {color: #6b7280; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Weighted Average Hybrid ARIMAX–LSTM Forecast Dashboard")
st.caption("Forecasts are shown in original level scale. ARIMAX is trained using log-transformed variables.")

# -----------------------------
# Helper functions
# -----------------------------
def make_template():
    next_month = LAST_HISTORY_DATE + pd.DateOffset(months=1)
    rows = []
    for i in range(3):
        d = next_month + pd.DateOffset(months=i)
        rows.append({"Year": d.year, "Month": d.month, "WTI_Price": "", "Exchange_Rate": ""})
    return pd.DataFrame(rows)

def validate_future_df(future_df):
    required = ["Year", "Month", "WTI_Price", "Exchange_Rate"]
    missing = [c for c in required if c not in future_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    future_df = future_df[required].copy()
    for c in required:
        future_df[c] = pd.to_numeric(future_df[c], errors="coerce")
    if future_df.isna().any().any():
        raise ValueError("All Year, Month, WTI_Price and Exchange_Rate values must be numeric.")

    future_df["Year"] = future_df["Year"].astype(int)
    future_df["Month"] = future_df["Month"].astype(int)
    if not future_df["Month"].between(1, 12).all():
        raise ValueError("Month must be between 1 and 12.")
    if (future_df[["WTI_Price", "Exchange_Rate"]] <= 0).any().any():
        raise ValueError("WTI_Price and Exchange_Rate must be positive values.")

    future_df["Date"] = pd.to_datetime(dict(year=future_df["Year"], month=future_df["Month"], day=1))
    future_df = future_df.sort_values("Date").drop_duplicates("Date")

    current_year = datetime.now().year
    min_allowed = LAST_HISTORY_DATE + pd.DateOffset(months=1)
    max_allowed = pd.Timestamp(year=current_year, month=12, day=1)

    if future_df["Date"].min() < min_allowed:
        raise ValueError(f"Forecast date must be after the last historical date: {LAST_HISTORY_DATE.strftime('%Y-%m')}")
    if future_df["Date"].max() > max_allowed:
        raise ValueError(f"Forecast is allowed only within 12 months of current year ({current_year}).")
    if len(future_df) > 12:
        raise ValueError("Maximum forecast horizon is 12 months.")

    return future_df.set_index("Date")

def recursive_forecast(future_input):
    future = future_input.copy()
    future["log_WTI_Price"] = np.log(future["WTI_Price"])
    future["log_Exchange_Rate"] = np.log(future["Exchange_Rate"])

    # ARIMAX multi-step forecast in log scale
    arimax_log = arimax_fit.forecast(steps=len(future), exog=future[EXOG_LOG])
    arimax_level = np.exp(np.asarray(arimax_log))

    # LSTM recursive forecast in log scale
    feature_history = history_df[FEATURE_COLS].copy()
    lstm_log_preds = []

    for dt in future.index:
        last_window = feature_history.iloc[-LOOKBACK:].copy()
        last_window.iloc[-1, last_window.columns.get_loc("log_WTI_Price")] = future.loc[dt, "log_WTI_Price"]
        last_window.iloc[-1, last_window.columns.get_loc("log_Exchange_Rate")] = future.loc[dt, "log_Exchange_Rate"]

        scaled_window = scaler.transform(last_window[FEATURE_COLS])
        X_next = scaled_window.reshape(1, LOOKBACK, len(FEATURE_COLS))
        pred_scaled = lstm_model.predict(X_next, verbose=0)[0, 0]

        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[0, 0] = pred_scaled
        pred_log = scaler.inverse_transform(dummy)[0, 0]
        lstm_log_preds.append(pred_log)

        new_row = pd.DataFrame({
            "log_Polymer_Import": [pred_log],
            "log_WTI_Price": [future.loc[dt, "log_WTI_Price"]],
            "log_Exchange_Rate": [future.loc[dt, "log_Exchange_Rate"]]
        }, index=[dt])
        feature_history = pd.concat([feature_history, new_row])

    lstm_level = np.exp(np.asarray(lstm_log_preds))
    hybrid_level = BEST_WEIGHT * lstm_level + (1 - BEST_WEIGHT) * arimax_level

    result = pd.DataFrame({
        "Date": future.index,
        "Year": future.index.year,
        "Month": future.index.month,
        "WTI_Price": future["WTI_Price"].values,
        "Exchange_Rate": future["Exchange_Rate"].values,
        "ARIMAX_Forecast": arimax_level,
        "LSTM_Forecast": lstm_level,
        "Hybrid_Forecast": hybrid_level
    })
    return result
from io import BytesIO

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Input_Template")
    return output.getvalue()
# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Model Information")
    st.write(f"Last historical month: **{LAST_HISTORY_DATE.strftime('%Y-%m')}**")
    st.write(f"LSTM weight: **{BEST_WEIGHT:.2f}**")
    st.write(f"ARIMAX weight: **{1-BEST_WEIGHT:.2f}**")
    st.write(f"Lookback: **{LOOKBACK} months**")
    st.divider()
    template_df = make_template()
    template_excel = convert_df_to_excel(template_df)

    st.download_button(
        label="📥 Download Excel Template",
        data=template_excel,
        file_name="forecast_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -----------------------------
# Input area
# -----------------------------
tab1, tab2 = st.tabs(["Manual Input: up to 3 months", "Excel Upload: more than 3 months"])

future_df_raw = None

with tab1:
    st.subheader("Manual Forecast Input")
    st.markdown("Enter Year, Month, WTI price and Exchange Rate. Manual input is limited to maximum three months.")
    n_months = st.selectbox("Number of months", [1, 2, 3], index=0)
    rows = []
    cols = st.columns(4)
    cols[0].markdown("**Year**")
    cols[1].markdown("**Month**")
    cols[2].markdown("**WTI Price**")
    cols[3].markdown("**Exchange Rate**")

    start_date = LAST_HISTORY_DATE + pd.DateOffset(months=1)
    for i in range(n_months):
        d = start_date + pd.DateOffset(months=i)
        c1, c2, c3, c4 = st.columns(4)
        year = c1.text_input(f"Year {i+1}", value=str(d.year), key=f"year_{i}")
        month = c2.text_input(f"Month {i+1}", value=str(d.month), key=f"month_{i}")
        wti = c3.text_input(f"WTI {i+1}", value="", key=f"wti_{i}")
        ex = c4.text_input(f"Exchange {i+1}", value="", key=f"ex_{i}")
        rows.append({"Year": year, "Month": month, "WTI_Price": wti, "Exchange_Rate": ex})

    if st.button("Forecast from Manual Input", type="primary"):
        future_df_raw = pd.DataFrame(rows)

with tab2:
    st.subheader("Excel Upload Forecast Input")
    st.markdown("Upload an Excel file with columns: `Year`, `Month`, `WTI_Price`, `Exchange_Rate`.")
    uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded is not None and st.button("Forecast from Excel Upload", type="primary"):
        future_df_raw = pd.read_excel(uploaded)

# -----------------------------
# Forecast output
# -----------------------------
if future_df_raw is not None:
    try:
        future_valid = validate_future_df(future_df_raw)
        forecast_result = recursive_forecast(future_valid)

        st.success("Forecast completed successfully.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast Months", len(forecast_result))
        c2.metric("Average Hybrid Forecast", f"{forecast_result['Hybrid_Forecast'].mean():,.0f}")
        c3.metric("Total Hybrid Forecast", f"{forecast_result['Hybrid_Forecast'].sum():,.0f}")

        display_result = forecast_result.copy()
        display_result["Date"] = display_result["Date"].dt.strftime("%Y-%m")
        for col in ["ARIMAX_Forecast", "LSTM_Forecast", "Hybrid_Forecast"]:
            display_result[col] = display_result[col].round(2)
        st.dataframe(display_result, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_df.index, y=history_df["Polymer_Import"], mode="lines", name="Historical Actual"))
        fig.add_trace(go.Scatter(x=forecast_result["Date"], y=forecast_result["Hybrid_Forecast"], mode="lines+markers", name="Hybrid Forecast"))
        fig.add_trace(go.Scatter(x=forecast_result["Date"], y=forecast_result["ARIMAX_Forecast"], mode="lines", name="ARIMAX Forecast"))
        fig.add_trace(go.Scatter(x=forecast_result["Date"], y=forecast_result["LSTM_Forecast"], mode="lines", name="LSTM Forecast"))
        fig.update_layout(
            title="Historical Polymer Import and Future Forecast",
            xaxis_title="Date",
            yaxis_title="Polymer Import Level",
            hovermode="x unified",
            template="plotly_white",
            height=520
        )
        st.plotly_chart(fig, use_container_width=True)

        output_bytes = forecast_result.to_excel(index=False, engine="openpyxl")
        st.download_button(
            "⬇️ Download Forecast Result Excel",
            data=output_bytes,
            file_name="hybrid_forecast_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(str(e))
else:
    st.info("Please enter manual input or upload an Excel file to generate forecasts.")

st.divider()
st.markdown("<span class='small-note'>Note: Forecast inputs must be within the current calendar year and after the last historical month.</span>", unsafe_allow_html=True)
