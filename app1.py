import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =====================================================
# Page config
# =====================================================

st.set_page_config(
    page_title="Polymer Import Forecasting",
    page_icon="📈",
    layout="wide"
)


# =====================================================
# Mobile-friendly CSS
# =====================================================

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.06);
    text-align: center;
}

@media only screen and (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    h1 {
        font-size: 1.6rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# Paths
# =====================================================

ARTIFACT_DIR = "artifacts"

HISTORY_PATH = os.path.join(ARTIFACT_DIR, "history_df.pkl")
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "hybrid_artifacts.pkl")
SCALER_X_PATH = os.path.join(ARTIFACT_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(ARTIFACT_DIR, "scaler_y.pkl")
LSTM_PATH = os.path.join(ARTIFACT_DIR, "lstm_model.keras")


# =====================================================
# Load artifacts
# =====================================================

@st.cache_resource
def load_artifacts():
    history_df = joblib.load(HISTORY_PATH)
    artifacts = joblib.load(ARTIFACT_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    lstm_model = load_model(LSTM_PATH)

    return history_df, artifacts, scaler_X, scaler_y, lstm_model


history_df, artifacts, scaler_X, scaler_y, lstm_model = load_artifacts()

best_order = artifacts["best_order"]
best_weight = artifacts["best_weight"]
lookback = artifacts["lookback"]
target_col = artifacts["target_col"]
lstm_features = artifacts["lstm_features"]
arimax_exog_cols = artifacts["arimax_exog_cols"]


# =====================================================
# Helper functions
# =====================================================

def prepare_future_df(input_df):
    df_future = input_df.copy()

    if "Date" not in df_future.columns:
        df_future["Date"] = pd.to_datetime(
            df_future["Year"].astype(str) + "-" +
            df_future["Month"].astype(str) + "-01"
        )
    else:
        df_future["Date"] = pd.to_datetime(df_future["Date"])

    df_future = df_future.sort_values("Date").reset_index(drop=True)

    df_future["log_WTI_Price"] = np.log(df_future["WTI_Price"])
    df_future["log_Exchange_Rate"] = np.log(df_future["Exchange_Rate"])

    return df_future


def validate_future_dates(df_future):
    current_year = datetime.now().year

    if len(df_future) > 12:
        st.error("Forecasting is allowed for maximum 12 months only.")
        st.stop()

    if not all(df_future["Date"].dt.year == current_year):
        st.error(f"Year is allowed only within the current year: {current_year}.")
        st.stop()

    if df_future["Date"].dt.month.max() > 12:
        st.error("Month must be within January to December.")
        st.stop()


def forecast_hybrid(future_df):
    temp_history = history_df.copy()
    results = []

    for i in range(len(future_df)):

        row = future_df.iloc[[i]]

        # -----------------------------
        # ARIMAX log forecast
        # -----------------------------
        arimax_model = SARIMAX(
            temp_history["log_y"],
            exog=temp_history[arimax_exog_cols],
            order=best_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        arimax_result = arimax_model.fit(disp=False)

        arimax_log_pred = arimax_result.forecast(
            steps=1,
            exog=row[arimax_exog_cols]
        )

        arimax_level_pred = np.exp(arimax_log_pred.iloc[0])

        # -----------------------------
        # Standalone LSTM forecast
        # -----------------------------
        recent = temp_history[lstm_features].iloc[-lookback:]

        recent_scaled = scaler_X.transform(recent)

        X_next = recent_scaled.reshape(1, lookback, len(lstm_features))

        lstm_scaled_pred = lstm_model.predict(X_next, verbose=0)

        lstm_level_pred = scaler_y.inverse_transform(lstm_scaled_pred)[0, 0]

        # -----------------------------
        # Weighted hybrid
        # -----------------------------
        hybrid_pred = (
            best_weight * lstm_level_pred
            + (1 - best_weight) * arimax_level_pred
        )

        results.append({
            "Date": row["Date"].iloc[0],
            "WTI_Price": row["WTI_Price"].iloc[0],
            "Exchange_Rate": row["Exchange_Rate"].iloc[0],
            "ARIMAX_Forecast": arimax_level_pred,
            "LSTM_Forecast": lstm_level_pred,
            "Weighted_Hybrid_Forecast": hybrid_pred
        })

        # Recursive update using predicted hybrid value
        new_row = pd.DataFrame({
            "WTI_Price": [row["WTI_Price"].iloc[0]],
            "Exchange_Rate": [row["Exchange_Rate"].iloc[0]],
            "Polymer_Import": [hybrid_pred],
            "log_y": [np.log(hybrid_pred)],
            "log_WTI_Price": [np.log(row["WTI_Price"].iloc[0])],
            "log_Exchange_Rate": [np.log(row["Exchange_Rate"].iloc[0])]
        }, index=[row["Date"].iloc[0]])

        temp_history = pd.concat([temp_history, new_row])

    return pd.DataFrame(results)


def create_template():
    current_year = datetime.now().year

    template = pd.DataFrame({
        "Year": [current_year, current_year, current_year],
        "Month": [1, 2, 3],
        "WTI_Price": [60, 65, 70],
        "Exchange_Rate": [3600, 3650, 3700]
    })

    return template


def retrain_model(new_data):
    new_data = new_data.copy()
    new_data["Date"] = pd.to_datetime(new_data["Date"])
    new_data = new_data.sort_values("Date").set_index("Date")

    new_data["log_y"] = np.log(new_data["Polymer_Import"])
    new_data["log_WTI_Price"] = np.log(new_data["WTI_Price"])
    new_data["log_Exchange_Rate"] = np.log(new_data["Exchange_Rate"])

    lstm_features = ["Polymer_Import", "WTI_Price", "Exchange_Rate"]
    arimax_exog_cols = ["log_WTI_Price", "log_Exchange_Rate"]

    scaler_X_new = MinMaxScaler()
    scaler_y_new = MinMaxScaler()

    scaled_X = scaler_X_new.fit_transform(new_data[lstm_features])
    scaled_y = scaler_y_new.fit_transform(new_data[["Polymer_Import"]])

    lookback = 12

    X, y = [], []

    for i in range(lookback, len(scaled_X)):
        X.append(scaled_X[i-lookback:i])
        y.append(scaled_y[i])

    X = np.array(X)
    y = np.array(y)

    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(lookback, 3)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    model.fit(
        X,
        y,
        epochs=200,
        batch_size=8,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    model.save(LSTM_PATH)
    joblib.dump(scaler_X_new, SCALER_X_PATH)
    joblib.dump(scaler_y_new, SCALER_Y_PATH)
    joblib.dump(new_data, HISTORY_PATH)

    updated_artifacts = {
        "best_order": best_order,
        "best_weight": best_weight,
        "lookback": lookback,
        "target_col": "Polymer_Import",
        "lstm_features": lstm_features,
        "arimax_exog_cols": arimax_exog_cols
    }

    joblib.dump(updated_artifacts, ARTIFACT_PATH)

    return True


# =====================================================
# UI
# =====================================================

st.title("📈 Polymer Import Forecasting Dashboard")
st.caption("Weighted Average Hybrid ARIMAX-LSTM Forecasting Model")

tab1, tab2, tab3 = st.tabs([
    "Manual Forecast",
    "Excel Forecast",
    "Retrain Model"
])


# =====================================================
# TAB 1: Manual Forecast
# =====================================================

with tab1:
    st.subheader("Manual Forecast Input")

    st.info("Manual input is allowed for maximum 3 months only: Jan, Feb, and Mar.")

    current_year = datetime.now().year

    manual_rows = []

    months = [
        ("Jan", 1),
        ("Feb", 2),
        ("Mar", 3)
    ]

    for month_name, month_num in months:
        st.markdown(f"### {month_name}")

        col1, col2, col3 = st.columns(3)

        with col1:
            year = st.text_input(
                f"Year for {month_name}",
                value=str(current_year),
                key=f"year_{month_name}"
            )

        with col2:
            wti = st.text_input(
                f"WTI Price for {month_name}",
                key=f"wti_{month_name}"
            )

        with col3:
            exchange = st.text_input(
                f"Exchange Rate for {month_name}",
                key=f"ex_{month_name}"
            )

        if wti and exchange:
            manual_rows.append({
                "Year": int(year),
                "Month": month_num,
                "WTI_Price": float(wti),
                "Exchange_Rate": float(exchange)
            })

    if st.button("Forecast Manual Input", use_container_width=True):
        if len(manual_rows) == 0:
            st.warning("Please enter at least one month of WTI and exchange rate.")
        else:
            input_df = pd.DataFrame(manual_rows)
            future_df = prepare_future_df(input_df)
            validate_future_dates(future_df)

            forecast_df = forecast_hybrid(future_df)

            st.success("Forecast completed.")
            st.dataframe(forecast_df, use_container_width=True)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df[target_col],
                mode="lines",
                name="Historical Polymer Import"
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Weighted_Hybrid_Forecast"],
                mode="lines+markers",
                name="Hybrid Forecast"
            ))

            fig.update_layout(
                title="Historical Series and Forecast",
                xaxis_title="Date",
                yaxis_title="Polymer Import",
                template="plotly_white",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            output = forecast_df.to_excel(index=False)

            st.download_button(
                label="Download Forecast Excel",
                data=forecast_df.to_csv(index=False).encode("utf-8"),
                file_name="manual_forecast_result.csv",
                mime="text/csv",
                use_container_width=True
            )


# =====================================================
# TAB 2: Excel Forecast
# =====================================================

with tab2:
    st.subheader("Excel Forecast Upload")

    st.write("Use Excel upload when forecasting more than 3 months.")

    template_df = create_template()

    st.download_button(
        label="Download Excel Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="forecast_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    uploaded_file = st.file_uploader(
        "Upload forecast input file",
        type=["xlsx", "csv"]
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)

        st.write("Uploaded Data")
        st.dataframe(input_df, use_container_width=True)

        if st.button("Forecast Uploaded File", use_container_width=True):
            future_df = prepare_future_df(input_df)
            validate_future_dates(future_df)

            forecast_df = forecast_hybrid(future_df)

            st.success("Forecast completed.")
            st.dataframe(forecast_df, use_container_width=True)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df[target_col],
                mode="lines",
                name="Historical Polymer Import"
            ))

            fig.add_trace(go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Weighted_Hybrid_Forecast"],
                mode="lines+markers",
                name="Hybrid Forecast"
            ))

            fig.update_layout(
                title="Historical Series and Forecast",
                xaxis_title="Date",
                yaxis_title="Polymer Import",
                template="plotly_white",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                label="Download Forecast Excel",
                data=forecast_df.to_csv(index=False).encode("utf-8"),
                file_name="excel_forecast_result.csv",
                mime="text/csv",
                use_container_width=True
            )


# =====================================================
# TAB 3: Retrain model
# =====================================================

with tab3:
    st.subheader("Retrain Model When New Data Is Available")

    st.warning(
        "Upload updated historical data with Date, WTI_Price, Exchange_Rate, and Polymer_Import."
    )

    retrain_file = st.file_uploader(
        "Upload updated historical dataset",
        type=["xlsx", "csv"],
        key="retrain_file"
    )

    if retrain_file is not None:
        if retrain_file.name.endswith(".csv"):
            new_data = pd.read_csv(retrain_file)
        else:
            new_data = pd.read_excel(retrain_file)

        st.dataframe(new_data.tail(), use_container_width=True)

        if st.button("Retrain Model", use_container_width=True):
            with st.spinner("Retraining model..."):
                retrain_model(new_data)

            st.success("Model retrained successfully. Please refresh the app.")