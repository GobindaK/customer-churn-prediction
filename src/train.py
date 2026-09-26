from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_generator import generate_customer_churn_dataset
from src.data_loader import DEFAULT_DATA_PATH, load_customer_data
from src.evaluate import compute_classification_metrics, save_confusion_matrix, save_model_comparison
from src.feature_engineering import engineer_features
from src.prediction import classify_risk
from src.preprocessing import clean_customer_data, profile_dataframe, preprocess_features
from src.utils import ensure_directory, save_json

MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
METRICS_DIR = RESULTS_DIR / 'metrics'
PREDICTIONS_DIR = RESULTS_DIR / 'predictions'


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns),
        ]
    )
    return transformer


def train_model(model_name: str, preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series):
    if model_name == 'logistic_regression':
        estimator = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    elif model_name == 'random_forest':
        estimator = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            min_samples_leaf=2,
        )
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', estimator)])
    pipeline.fit(X_train, y_train)
    return pipeline


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series):
    preprocessor = build_preprocessor(X_train)
    estimator = RandomForestClassifier(random_state=42, class_weight='balanced')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', estimator)])
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 6, 10],
        'model__min_samples_leaf': [1, 2],
    }
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def save_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    ensure_directory(MODELS_DIR)
    ensure_directory(FIGURES_DIR)
    ensure_directory(METRICS_DIR)
    ensure_directory(PREDICTIONS_DIR)

    if not DEFAULT_DATA_PATH.exists():
        generate_customer_churn_dataset(DEFAULT_DATA_PATH)

    raw_df = load_customer_data(DEFAULT_DATA_PATH)
    cleaned_df = clean_customer_data(raw_df)
    profiled = profile_dataframe(cleaned_df)
    save_json(profiled, METRICS_DIR / 'data_profile.json')

    engineered = engineer_features(cleaned_df)
    X, y = preprocess_features(engineered)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {}
    models['logistic_regression'] = train_model('logistic_regression', build_preprocessor(X_train), X_train, y_train)
    models['random_forest'] = train_model('random_forest', build_preprocessor(X_train), X_train, y_train)
    models['random_forest_tuned'] = tune_random_forest(X_train, y_train)

    comparison_rows = []
    for model_name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        comparison_rows.append({'model': model_name, **metrics})

        save_confusion_matrix(y_test, y_pred, FIGURES_DIR / f'{model_name}_confusion_matrix.png')
        save_roc_curve(model, X_test, y_test, FIGURES_DIR / f'{model_name}_roc_curve.png')

        model_path = MODELS_DIR / f'{model_name}.joblib'
        joblib.dump(model, model_path)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(METRICS_DIR / 'model_comparison.csv', index=False)
    save_model_comparison(
        [
            compute_classification_metrics(y_test, models['logistic_regression'].predict(X_test), models['logistic_regression'].predict_proba(X_test)[:, 1]),
            compute_classification_metrics(y_test, models['random_forest'].predict(X_test), models['random_forest'].predict_proba(X_test)[:, 1]),
            compute_classification_metrics(y_test, models['random_forest_tuned'].predict(X_test), models['random_forest_tuned'].predict_proba(X_test)[:, 1]),
        ],
        ['logistic_regression', 'random_forest', 'random_forest_tuned'],
        METRICS_DIR / 'model_metrics.csv',
    )

    model_metrics = [
        {'model': 'logistic_regression', **compute_classification_metrics(y_test, models['logistic_regression'].predict(X_test), models['logistic_regression'].predict_proba(X_test)[:, 1])},
        {'model': 'random_forest', **compute_classification_metrics(y_test, models['random_forest'].predict(X_test), models['random_forest'].predict_proba(X_test)[:, 1])},
        {'model': 'random_forest_tuned', **compute_classification_metrics(y_test, models['random_forest_tuned'].predict(X_test), models['random_forest_tuned'].predict_proba(X_test)[:, 1])},
    ]
    selected_model_name = max(model_metrics, key=lambda row: (row['f1'], row['roc_auc']))['model']
    final_model = models[selected_model_name]
    final_probs = final_model.predict_proba(X_test)[:, 1]
    final_risk = classify_risk(final_probs)
    predictions = pd.DataFrame(
        {
            'customer_id': X_test['customer_id'] if 'customer_id' in X_test.columns else X_test.index.astype(str),
            'churn_probability': final_probs,
            'risk_band': final_risk,
            'actual_churn': y_test.to_numpy(),
        }
    )
    predictions.to_csv(PREDICTIONS_DIR / 'model_predictions.csv', index=False)

    summary = {
        'dataset_rows': int(len(cleaned_df)),
        'dataset_columns': int(cleaned_df.shape[1]),
        'target_distribution': y.value_counts(normalize=True).to_dict(),
        'final_model': selected_model_name,
        'risk_thresholds': {'LOW': 0.30, 'MEDIUM': 0.60},
    }
    save_json(summary, METRICS_DIR / 'project_summary.json')
    return summary


if __name__ == '__main__':
    main()
