import pandas as pd

from src.data_loader import DEFAULT_DATA_PATH, load_customer_data
from src.preprocessing import preprocess_features
from src.prediction import classify_risk


def test_load_customer_data_returns_dataframe():
    df = load_customer_data(DEFAULT_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_preprocess_features_returns_expected_columns():
    df = pd.DataFrame(
        {
            'customer_id': ['C1', 'C2', 'C3'],
            'gender': ['Male', 'Female', None],
            'tenure': [12, 5, 24],
            'monthly_charges': [50, 80, 60],
            'contract_type': ['Month-to-month', 'One year', 'Two year'],
            'churn': [1, 0, 0],
        }
    )
    X, y = preprocess_features(df)
    assert 'tenure' in X.columns
    assert 'contract_type' in X.columns
    assert len(y) == len(df)


def test_classify_risk_returns_valid_bands():
    probs = [0.1, 0.5, 0.8]
    result = classify_risk(probs)
    assert set(result) == {'LOW', 'MEDIUM', 'HIGH'}
    assert all(0.0 <= p <= 1.0 for p in probs)
