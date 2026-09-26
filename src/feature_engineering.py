from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a small set of meaningful, non-leaky features for customer churn modeling."""
    engineered = df.copy()

    if 'tenure_months' in engineered.columns and 'monthly_charges' in engineered.columns:
        engineered['monthly_to_tenure_ratio'] = engineered['monthly_charges'] / np.maximum(engineered['tenure_months'], 1)
        engineered['customer_value'] = engineered['monthly_charges'] * engineered['tenure_months']

    if 'support_contacts' in engineered.columns and 'tenure_months' in engineered.columns:
        engineered['support_contact_rate'] = engineered['support_contacts'] / np.maximum(engineered['tenure_months'], 1)

    if 'avg_usage_hours' in engineered.columns:
        engineered['usage_above_baseline'] = engineered['avg_usage_hours'] - engineered['avg_usage_hours'].median()

    if 'satisfaction_score' in engineered.columns and 'support_contacts' in engineered.columns:
        engineered['satisfaction_support_gap'] = engineered['satisfaction_score'] - engineered['support_contacts']

    if 'days_since_last_login' in engineered.columns:
        engineered['inactive_customer_flag'] = (engineered['days_since_last_login'] > engineered['days_since_last_login'].median()).astype(int)

    return engineered
