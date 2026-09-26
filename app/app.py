from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.data_loader import load_customer_data
from src.feature_engineering import engineer_features
from src.prediction import classify_risk
from src.preprocessing import clean_customer_data
from src.train import MODELS_DIR, PREDICTIONS_DIR, __file__ as _train_file


st.set_page_config(page_title='Customer Churn Dashboard', layout='wide')
st.title('Customer Churn Prediction Dashboard')

try:
    data = load_customer_data()
    cleaned = clean_customer_data(data)
    X = engineer_features(cleaned)
    predictions = pd.read_csv(PREDICTIONS_DIR / 'model_predictions.csv')
    total_customers = len(predictions)
    high_risk = (predictions['risk_band'] == 'HIGH').sum()
    medium_risk = (predictions['risk_band'] == 'MEDIUM').sum()
    low_risk = (predictions['risk_band'] == 'LOW').sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total customers', total_customers)
    col2.metric('High risk', high_risk)
    col3.metric('Medium risk', medium_risk)
    col4.metric('Low risk', low_risk)

    st.subheader('Risk distribution')
    st.bar_chart(predictions['risk_band'].value_counts())

    customer_id = st.selectbox('Select customer', predictions['customer_id'])
    selected = predictions[predictions['customer_id'] == customer_id].iloc[0]
    st.write(f"Customer ID: {customer_id}")
    st.write(f"Churn probability: {selected['churn_probability']:.3f}")
    st.write(f"Risk band: {selected['risk_band']}")

except Exception as exc:  # pragma: no cover
    st.warning(f'Model output is not available yet. Run the training script first. Details: {exc}')
