# SoCal House Price Streamlit App

This deployment turns `socal2 (1).ipynb` into an interactive Streamlit dashboard.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What is included

- EDA dashboard for city, street, sqft, bedrooms, and bathrooms.
- Random Forest tabular price predictor.
- Property image gallery from `socal2/socal_pics`.
- Image + tabular deep learning design explanation.
- HeroUI-inspired Streamlit styling through custom CSS.

## Deployment notes

The app trains a cached Random Forest model at startup from `socal2.csv`.
The notebook's EfficientNet model is documented in the app, but it is not deployed because no saved `.keras` model artifact is present in the workspace.
