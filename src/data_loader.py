from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_generator import generate_customer_churn_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'customer_churn_dataset.csv'
SAMPLE_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'sample_customer_churn.csv'


def load_customer_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the customer churn dataset from disk."""
    file_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    file_path = file_path if file_path.is_absolute() else PROJECT_ROOT / file_path

    if not file_path.exists():
        if file_path.name == 'sample_customer_churn.csv' or file_path.name == 'customer_churn_dataset.csv':
            generate_customer_churn_dataset(file_path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {file_path}. "
                'No dataset was included in the repository, so the project uses a synthetic churn dataset. '
                'Generate it with the training script or add your own CSV file.'
            )

    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f'Dataset at {file_path} is empty.')
    return df
