# =====================================================
# app1.py
# Weighted Average Hybrid ARIMAX-LSTM Streamlit App
# =====================================================

import os
from io import BytesIO
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =====================================================
# Streamlit page setting
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
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

.stButton button {
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
}

@media only screen and (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1 {
        font-size: 1.5rem !important;
    }

    h2, h3 {
        font-size: 1.1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# Artifact paths
# =====================================================

ARTIFACT_DIR = "artifacts"

HISTORY_PATH = os.path.join(ARTIFACT_DIR, "history_df.pkl")
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "hybrid_artifacts.pkl")
SCALER_X_PATH = os.path.join(ARTIFACT_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(ARTIFACT_DIR, "scaler_y.pkl")
LSTM_PATH = os.path.join(ARTIFACT_DIR, "lstm_model.keras")


# =====================================================
# Load saved model artifacts
# =====================================================

@st.cache_resource
def load_artifacts():
    history_df = joblib.load(HISTORY_PATH)
    artifacts = joblib.load(ARTIFACT_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    lstm_model = load_model(LSTM_PATH)

    return history_df, artifacts, scaler_X, scaler_y, lstm_model


try:
    history_df, artifacts, scaler_X, scaler_y, lstm_model = load_artifacts()
except Exception as e:
    st.error("Model artifacts could not be loaded. Please check the artifacts folder.")
    st.exception(e)
    st.stop()


best_order = artifacts["best_order"]
best_weight = artifacts["best_weight"]
lookback = artifacts["lookback"]
target_col = artifacts["target_col"]
lstm_features = artifacts["lstm_features"]
arimax_exog_cols = artifacts["arimax_exog_cols"]


# =====================================================
# Helper functions
# =====================================================

def dataframe_to_excel_bytes(df, sheet_name="Forecast_Result"):
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

    buffer.seek(0)
    return buffer


def prepare_future_df(input_df):
    df_future = input_df.copy()

    if "Date" not in df_future.columns:
        df_future["Date"] = pd.to_datetime(
            df_future["Year"].astype(str)
            + "-"
            + df_future["Month"].astype(str)
            + "-01"
        )
    else:
        df_future["Date"] = pd.to_datetime(df_future["Date"])

    df_future = df_future.sort_values("Date").reset_index(drop=True)

    df_future["WTI_Price"] = pd.to_numeric(df_future["WTI_Price"])
    df_future["Exchange_Rate"] = pd.to_numeric(df_future["Exchange_Rate"])

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

    if df_future["Date"].duplicated().any():
        st.error("Duplicate months are not allowed.")
        st.stop()

    if (df_future["WTI_Price"] <= 0).any() or (df_future["Exchange_Rate"] <= 0).any():
        st.error("WTI Price and Exchange Rate must be greater than zero.")
        st.stop()


def forecast_hybrid(future_df):
    temp_history = history_df.copy()
    results = []

    for i in range(len(future_df)):
        row = future_df.iloc[[i]]

        # -----------------------------
        # ARIMAX forecast in log scale
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

        arimax_level_pred = float(np.exp(arimax_log_pred.iloc[0]))

        # -----------------------------
        # Standalone LSTM forecast
        # -----------------------------
        recent = temp_history[lstm_features].iloc[-lookback:]

        recent_scaled = scaler_X.transform(recent)

        X_next = recent_scaled.reshape(
            1,
            lookback,
            len(lstm_features)
        )

        lstm_scaled_pred = lstm_model.predict(X_next, verbose=0)

        lstm_level_pred = float(
            scaler_y.inverse_transform(lstm_scaled_pred)[0, 0]
        )

        # -----------------------------
        # Weighted Average Hybrid
        # -----------------------------
        hybrid_pred = (
            best_weight * lstm_level_pred
            + (1 - best_weight) * arimax_level_pred
        )

        results.append({
            "Date": row["Date"].iloc[0],
            "WTI_Price": row["WTI_Price"].iloc[0],
            "Exchange_Rate": row["Exchange_Rate"].iloc[0],
            "ARIMAX_Forecast": round(arimax_level_pred, 2),
            "LSTM_Forecast": round(lstm_level_pred, 2),
            "Weighted_Hybrid_Forecast": round(hybrid_pred, 2)
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

    template_df = pd.DataFrame({
        "Year": [current_year] * 12,
        "Month": list(range(1, 13)),
        "WTI_Price": [60] * 12,
        "Exchange_Rate": [3600] * 12
    })

    return template_df


def plot_forecast(forecast_df):
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
        title="Historical Polymer Import and Forecast",
        xaxis_title="Date",
        yaxis_title="Polymer Import",
        template="plotly_white",
        height=520,
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)


def retrain_model(new_data, selected_order):
    new_data = new_data.copy()

    required_cols = ["Date", "WTI_Price", "Exchange_Rate", "Polymer_Import"]

    missing_cols = [col for col in required_cols if col not in new_data.columns]

    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()

    new_data["Date"] = pd.to_datetime(new_data["Date"])
    new_data = new_data.sort_values("Date").set_index("Date")

    new_data["WTI_Price"] = pd.to_numeric(new_data["WTI_Price"])
    new_data["Exchange_Rate"] = pd.to_numeric(new_data["Exchange_Rate"])
    new_data["Polymer_Import"] = pd.to_numeric(new_data["Polymer_Import"])

    new_data["log_y"] = np.log(new_data["Polymer_Import"])
    new_data["log_WTI_Price"] = np.log(new_data["WTI_Price"])
    new_data["log_Exchange_Rate"] = np.log(new_data["Exchange_Rate"])

    lstm_features_new = ["Polymer_Import", "WTI_Price", "Exchange_Rate"]
    arimax_exog_cols_new = ["log_WTI_Price", "log_Exchange_Rate"]

    scaler_X_new = MinMaxScaler()
    scaler_y_new = MinMaxScaler()

    scaled_X = scaler_X_new.fit_transform(new_data[lstm_features_new])
    scaled_y = scaler_y_new.fit_transform(new_data[["Polymer_Import"]])

    X, y = [], []

    for i in range(lookback, len(scaled_X)):
        X.append(scaled_X[i-lookback:i])
        y.append(scaled_y[i])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        st.error("Not enough data for retraining. At least more than 12 months are required.")
        st.stop()

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
        epochs=100,
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
    
# =====================================================
# Recalculate Hybrid Weight
# =====================================================
    candidate_weights = np.arange(0, 1.01, 0.01)
    weight_results = []
    validation_size = int(len(new_data) * 0.2)
    validation_df = new_data.iloc[-validation_size:].copy()
    history_temp = new_data.iloc[:-validation_size].copy()

    arimax_preds = []
    lstm_preds = []
    actuals = []

    for i in range(len(validation_df)):

        row = validation_df.iloc[[i]]

    # -----------------------------------------
    # ARIMAX Forecast
    # -----------------------------------------

        arimax_model = SARIMAX(
        history_temp["log_y"],
        exog=history_temp[arimax_exog_cols_new],
        order=selected_order,
        enforce_stationarity=False,
        enforce_invertibility=False
        )

        arimax_result = arimax_model.fit(disp=False)

        arimax_log_pred = arimax_result.forecast(
        steps=1,
        exog=row[arimax_exog_cols_new]
        )

        arimax_level_pred = np.exp(arimax_log_pred.iloc[0])

    # -----------------------------------------
    # LSTM Forecast
    # -----------------------------------------

        recent = history_temp[lstm_features_new].iloc[-lookback:]

        recent_scaled = scaler_X_new.transform(recent)

        X_next = recent_scaled.reshape(
            1,
            lookback,
            len(lstm_features_new)
            )

        lstm_scaled_pred = model.predict(X_next, verbose=0)

        lstm_level_pred = scaler_y_new.inverse_transform(
        lstm_scaled_pred
            )[0, 0]

        arimax_preds.append(arimax_level_pred)
        lstm_preds.append(lstm_level_pred)
        actuals.append(row["Polymer_Import"].iloc[0])

        history_temp = pd.concat([history_temp, row])

# =====================================================
# Search Best Hybrid Weight
# =====================================================

        for w in candidate_weights:

            hybrid_preds = (
            w * np.array(lstm_preds)
            + (1 - w) * np.array(arimax_preds)
                )

            rmse = np.sqrt(np.mean((np.array(actuals) - hybrid_preds) ** 2))

            weight_results.append({
                "weight": w,
                "rmse": rmse
                })

            weight_df = pd.DataFrame(weight_results)

            new_best_weight = float( weight_df.sort_values("rmse").iloc[0]["weight"] )

            print("Updated Best Weight:", new_best_weight)
    updated_artifacts = {
    "best_order": selected_order,
    "best_weight": new_best_weight,
    "lookback": lookback,
    "target_col": "Polymer_Import",
    "lstm_features": lstm_features_new,
    "arimax_exog_cols": arimax_exog_cols_new
    }
    joblib.dump(updated_artifacts, ARTIFACT_PATH)
    return True

# =====================================================
# Main UI
# =====================================================

st.title("📈 Polymer Import Forecasting Dashboard")
st.caption("Weighted Average Hybrid ARIMAX-LSTM Model")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Best LSTM Weight", round(best_weight, 2))

with col_b:
    st.metric("Best ARIMAX Weight", round(1 - best_weight, 2))

with col_c:
    st.metric("ARIMAX Order", str(best_order))


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

    st.info(
        "Manual input allows maximum 3 months. "
        "Please select months from Jan to Dec. "
        "Months must be in order and duplicate months are not allowed."
    )

    current_year = datetime.now().year

    month_options = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12
    }

    num_months = st.selectbox(
        "Number of months to forecast",
        options=[1, 2, 3],
        index=0
    )

    manual_rows = []
    selected_month_nums = []

    for i in range(num_months):
        st.markdown(f"### Forecast Input {i + 1}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            year = st.text_input(
                f"Year {i + 1}",
                value=str(current_year),
                key=f"manual_year_{i}"
            )

        with col2:
            month_name = st.selectbox(
                f"Month {i + 1}",
                options=list(month_options.keys()),
                key=f"manual_month_{i}"
            )

        with col3:
            wti = st.text_input(
                f"WTI Price {i + 1}",
                key=f"manual_wti_{i}"
            )

        with col4:
            exchange = st.text_input(
                f"Exchange Rate {i + 1}",
                key=f"manual_exchange_{i}"
            )

        month_num = month_options[month_name]
        selected_month_nums.append(month_num)

        if wti.strip() != "" and exchange.strip() != "":
            try:
                manual_rows.append({
                    "Year": int(year),
                    "Month": month_num,
                    "WTI_Price": float(wti),
                    "Exchange_Rate": float(exchange)
                })
            except ValueError:
                st.error(f"Please enter valid numeric values for Forecast Input {i + 1}.")

    # -----------------------------
    # Month validation
    # -----------------------------

    if len(selected_month_nums) != len(set(selected_month_nums)):
        st.error("Duplicate months are not allowed. Please select different months.")

    if selected_month_nums != sorted(selected_month_nums):
        st.error("Months must be selected in chronological order, for example Jan, Feb, Mar.")

    if st.button("Forecast Manual Input", use_container_width=True):

        if len(selected_month_nums) != len(set(selected_month_nums)):
            st.error("Duplicate months are not allowed.")
            st.stop()

        if selected_month_nums != sorted(selected_month_nums):
            st.error("Months must be selected in chronological order.")
            st.stop()

        if len(manual_rows) != num_months:
            st.warning("Please complete WTI Price and Exchange Rate for all selected months.")
            st.stop()

        input_df = pd.DataFrame(manual_rows)

        future_df = prepare_future_df(input_df)
        validate_future_dates(future_df)

        with st.spinner("Forecasting..."):
            forecast_df = forecast_hybrid(future_df)

        st.success("Forecast completed.")

        st.dataframe(forecast_df, use_container_width=True)

        plot_forecast(forecast_df)

        excel_buffer = dataframe_to_excel_bytes(
            forecast_df,
            sheet_name="Manual_Forecast"
        )

        st.download_button(
            label="Download Forecast Excel",
            data=excel_buffer,
            file_name="manual_forecast_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
# =====================================================
# TAB 2: Excel Forecast
# =====================================================

with tab2:
    st.subheader("Excel Forecast Upload")

    st.write("Use Excel upload when forecasting more than 3 months.")

    template_df = create_template()

    template_buffer = dataframe_to_excel_bytes(
        template_df,
        sheet_name="Forecast_Template"
    )

    st.download_button(
        label="Download Excel Template",
        data=template_buffer,
        file_name="forecast_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    uploaded_file = st.file_uploader(
        "Upload forecast input file",
        type=["xlsx", "csv"],
        key="forecast_upload"
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

            with st.spinner("Forecasting..."):
                forecast_df = forecast_hybrid(future_df)

            st.success("Forecast completed.")

            st.dataframe(forecast_df, use_container_width=True)

            plot_forecast(forecast_df)

            excel_buffer = dataframe_to_excel_bytes(
                forecast_df,
                sheet_name="Excel_Forecast"
            )

            st.download_button(
                label="Download Forecast Excel",
                data=excel_buffer,
                file_name="excel_forecast_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


# =====================================================
# TAB 3: Retrain Model
# =====================================================

with tab3:
    st.subheader("Retrain Model When New Data Is Available")

    st.warning(
        "Upload updated historical data with Date, WTI_Price, Exchange_Rate, and Polymer_Import."
    )
    # =====================================================
    # ARIMAX order input
    # =====================================================

    st.markdown("### ARIMAX Order Selection")

    col1, col2, col3 = st.columns(3)

    with col1:
        retrain_p = st.number_input(
            "AR Order (p)",
            min_value=0,
            max_value=5,
            value=int(best_order[0]),
            step=1
            )

    with col2:
        retrain_d = st.number_input(
            "Differencing Order (d)",
            min_value=0,
            max_value=2,
            value=int(best_order[1]),
            step=1
            )

    with col3:
        retrain_q = st.number_input(
            "MA Order (q)",
            min_value=0,
            max_value=5,
            value=int(best_order[2]),
            step=1
            )

    selected_order = (
        int(retrain_p),
        int(retrain_d),
        int(retrain_q))

    st.info(f"Selected ARIMAX Order: {selected_order}")
    retrain_file = st.file_uploader(
        "Upload updated historical dataset",
        type=["xlsx", "csv"],
        key="retrain_upload")

    if retrain_file is not None:
        if retrain_file.name.endswith(".csv"):
            new_data = pd.read_csv(retrain_file)
        else:
            new_data = pd.read_excel(retrain_file)

        st.write("Preview of uploaded data")
        st.dataframe(new_data.tail(), use_container_width=True)

        if st.button("Retrain Model", use_container_width=True):
            with st.spinner("Retraining model..."):
                retrain_model(new_data, selected_order)

            st.success("Model retrained successfully.")

            st.cache_resource.clear()

            st.rerun()
         
