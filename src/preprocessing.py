from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

TARGET_COLUMN = 'churn'


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Return a compact profile of the dataset."""
    profile = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'column_names': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isna().sum().to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
        'descriptive_statistics': df.describe(include='all').to_dict(),
        'churn_distribution': df[TARGET_COLUMN].value_counts(normalize=True).to_dict() if TARGET_COLUMN in df.columns else {},
    }
    return profile


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple, explainable cleaning for academic churn modeling."""
    cleaned = df.copy()

    cleaned['customer_id'] = cleaned['customer_id'].astype(str)
    cleaned['churn'] = pd.to_numeric(cleaned.get('churn', 0), errors='coerce').fillna(0).astype(int)

    for column in cleaned.columns:
        if column == 'customer_id':
            continue
        if cleaned[column].dtype.kind in 'ifb':
            cleaned[column] = pd.to_numeric(cleaned[column], errors='coerce')
            median_value = cleaned[column].median()
            if pd.notna(median_value):
                cleaned[column] = cleaned[column].fillna(median_value)
        else:
            cleaned[column] = cleaned[column].astype(str).str.strip()
            cleaned[column] = cleaned[column].replace({'nan': pd.NA, 'None': pd.NA, 'NaN': pd.NA})
            mode_value = cleaned[column].mode(dropna=True)
            if not mode_value.empty:
                cleaned[column] = cleaned[column].fillna(mode_value.iloc[0])

    cleaned = cleaned.drop_duplicates(subset=['customer_id'], keep='first').reset_index(drop=True)
    return cleaned


def preprocess_features(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target series y after cleaning and removing leakage columns."""
    cleaned = clean_customer_data(df)
    leakage_columns = {'customer_id', target_column}
    X = cleaned.drop(columns=[column for column in leakage_columns if column in cleaned.columns], errors='ignore')
    y = cleaned[target_column]
    return X, y
