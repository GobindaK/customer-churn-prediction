from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_customer_churn_dataset(output_path: str | Path | None = None) -> pd.DataFrame:
    """Generate a synthetic but realistic customer churn dataset for academic use."""
    rng = np.random.default_rng(42)
    n_rows = 1200

    customer_ids = [f'C{idx:04d}' for idx in range(1, n_rows + 1)]
    genders = rng.choice(['Male', 'Female', 'Other'], size=n_rows)
    age = rng.integers(18, 75, size=n_rows)
    tenure_months = rng.integers(1, 72, size=n_rows)
    monthly_charges = rng.normal(62, 18, size=n_rows)
    monthly_charges = np.clip(monthly_charges, 20, 150)
    contract_type = rng.choice(['Month-to-month', 'One year', 'Two year'], size=n_rows, p=[0.55, 0.30, 0.15])
    payment_method = rng.choice(['Electronic check', 'Bank transfer', 'Credit card', 'Mailed check'], size=n_rows, p=[0.40, 0.25, 0.20, 0.15])
    support_contacts = rng.poisson(1.5, size=n_rows)
    avg_usage_hours = rng.normal(32, 18, size=n_rows)
    avg_usage_hours = np.clip(avg_usage_hours, 5, 120)
    satisfaction_score = rng.integers(1, 6, size=n_rows)
    days_since_last_login = rng.integers(1, 90, size=n_rows)

    churn_probability = (
        0.45
        + (support_contacts / 6.0) * 0.15
        + (days_since_last_login / 90.0) * 0.18
        + ((30 - satisfaction_score) / 15.0) * 0.12
        + (np.where(contract_type == 'Month-to-month', 1, 0) * 0.20)
        + (np.where(payment_method == 'Electronic check', 1, 0) * 0.10)
        - (tenure_months / 72.0) * 0.22
        - (avg_usage_hours / 120.0) * 0.08
    )
    churn_probability = np.clip(churn_probability, 0.02, 0.90)
    churn = rng.binomial(1, churn_probability)

    df = pd.DataFrame(
        {
            'customer_id': customer_ids,
            'gender': genders,
            'age': age,
            'tenure_months': tenure_months,
            'monthly_charges': np.round(monthly_charges, 2),
            'contract_type': contract_type,
            'payment_method': payment_method,
            'support_contacts': support_contacts,
            'avg_usage_hours': np.round(avg_usage_hours, 2),
            'satisfaction_score': satisfaction_score,
            'days_since_last_login': days_since_last_login,
            'churn': churn,
        }
    )

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

    return df
