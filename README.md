# Mizal Analytics: Elite Talent Predictor

Mizal Analytics is a Streamlit predictive analytics system for identifying employees who are highly suitable for high-priority task assignments. The app lets users evaluate existing employees, add new employee records, generate suitability predictions, view an elite roster, and inspect model analytics.

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application and prediction system |
| `structured_data (1).csv` | Local employee performance dataset |
| `Mizal_Analytics_Model_Development.ipynb` | Notebook for preprocessing, model training, evaluation, and feature importance |
| `requirements.txt` | Python dependencies for running and deploying the app |

## Model Summary

The system predicts whether an employee belongs to the `High Suitability` group for important task assignments. The target variable is created from a composite score using six indicators:

| Feature | Weight |
|---|---:|
| average_task_quality | 30% |
| tasks_completed | 20% |
| projects_led | 15% |
| deadline_met_score | 15% |
| client_satisfaction_score | 10% |
| efficiency_score | 10% |

The final model is a Random Forest Classifier with refined hyperparameters. The final out-of-bag accuracy is approximately 97.31% on the current cleaned dataset.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Use these settings:

| Setting | Value |
|---|---|
| Repository | `KayeMizal/mizal_analytics` |
| Branch | `main` |
| Main file path | `app.py` |

After deployment, copy the public Streamlit app URL and submit it as the deployed application link.
