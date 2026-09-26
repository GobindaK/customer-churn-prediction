# AI-Powered Customer Churn Prediction and Intelligent Decision Support Platform

## 1. Project title
AI-Powered Customer Churn Prediction and Intelligent Decision Support Platform

## 2. Project objective
This project builds a simple academic churn prediction workflow for customer retention analysis. It loads the dataset, cleans and profiles it, explores the relationships between customer behaviour and churn, trains a baseline logistic regression model and a random forest model, evaluates them, and produces churn probabilities and risk bands for decision support.

## 3. Business problem
The business wants to identify customers likely to churn early, so the retention team can prioritize intervention and reduce customer loss. The prediction target is the churn indicator in the dataset. One row is one customer.

## 4. Dataset
No dataset was present in the repository at project start. The project therefore uses a synthetic churn dataset generated locally in the training workflow so that the code can run reproducibly. The dataset is explicitly marked as synthetic and is stored under `data/raw/`.

## 5. Technologies
- Python 3.13
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- streamlit
- pytest

## 6. Project structure

```text
customer-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── prediction.py
│   └── utils.py
├── models/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── app/
│   └── app.py
├── tests/
└── .venv/
```

## 7. Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 8. How to run

```bash
python src/train.py
python -m pytest -q
streamlit run app/app.py
```

## 9. Data preprocessing
The project includes a data-loading step and a compact cleaning layer that handles missing values, type coercion, duplicate rows, and invalid values. It is designed to be transparent and easy to explain in an MCA viva.

## 10. EDA
The project includes exploratory summaries and plots for churn distribution and key feature relationships. These reports are generated during the training process.

## 11. Feature engineering
The project creates a small number of engineered features such as ratios and behavioural indicators when relevant columns are available.

## 12. Models
- Logistic Regression
- Random Forest
- Model comparison with actual evaluation metrics

## 13. Model evaluation
The model evaluation uses accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix output.

## 14. Explainability
The explainability layer uses feature importance from the selected model and a simple per-customer explanation summary.

## 15. Risk prediction
Churn probabilities are converted into LOW, MEDIUM, and HIGH risk bands.

## 16. Dashboard
The application in `app/app.py` provides a simple local Streamlit dashboard for customer risk review.

## 17. Responsible AI
This project acknowledges privacy, bias, explainability, and human oversight. It does not automate final customer decisions without human review.

## 18. Deployment design
The project includes a simple documented deployment design rather than a cloud-native production architecture.

## 19. Monitoring
Monitoring should track data drift, prediction drift, and model performance once actual churn outcomes become available.

## 20. Limitations
- No original dataset was supplied in the repository.
- The current dataset is synthetic for reproducibility.
- This is an academic prototype, not a production-grade enterprise system.

## 21. Future improvements
- Add a real customer churn dataset from a trusted source.
- Add more extensive feature engineering.
- Add calibration and threshold tuning.
- Recruit domain feedback from the business team.

## 22. Implemented vs future enhancements

Implemented:
- project structure
- synthetic data generation
- data loading and cleaning
- preprocessing utilities
- model training and evaluation
- risk classification
- tests
- Streamlit app

Future enhancement:
- real-world dataset integration
- richer model tuning and monitoring infrastructure
- production deployment and governance workflow
