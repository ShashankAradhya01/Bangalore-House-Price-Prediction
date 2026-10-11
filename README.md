# Bangalore House Price Prediction

A machine learning project that predicts residential property prices
in Bangalore using property details such as location, area, BHK,
bathrooms, balcony count, and area type.

## Features

- Data cleaning and exploratory data analysis
- Feature engineering for housing data
- Machine learning using a Random Forest regression pipeline
- FastAPI REST API for price predictions
- Streamlit web interface for user-friendly predictions
- Saved model pipeline for reuse

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- FastAPI and Uvicorn
- Streamlit
- Joblib

## Project Structure

```text
Bangalore_House_Price_Prediction/
├── data/
├── models/
│   └── house_price_pipeline.joblib
├── notebooks/
│   └── 01_Data_Exploration.ipynb
├── house_api.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment.

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

4. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the API

Open a terminal in the project folder:

```bash
python -m uvicorn house_api:app --host 127.0.0.1 --port 8000
```

API documentation:

http://127.0.0.1:8000/docs

## Run the Streamlit Application

Open another terminal in the same project folder:

```bash
python -m streamlit run streamlit_app.py
```

The Streamlit app uses the API at `http://127.0.0.1:8000`.
Start the API before making predictions in the app.

## Model Evaluation

The current evaluation run reported:

- MAE: 35.49 lakhs
- RMSE: 94.98 lakhs
- R² score: 0.6361

These results are from a reconstructed test split and should be
verified against the original training split before being treated
as a definitive benchmark. Predictions are estimates, not property
valuations.

## Limitations

Predictions depend on the quality and coverage of the training data.
Market prices change over time, and the model may not capture every
property-specific factor.

## Future Improvements

- Tune model hyperparameters
- Improve outlier handling and location features
- Add prediction visualizations
- Deploy the application online
- Add automated model evaluation

## Disclaimer

This project is for educational purposes. Predictions should not be
used as the sole basis for property purchase or investment decisions.
