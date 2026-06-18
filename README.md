# SoCal House Price Prediction

A machine learning project for analyzing and predicting Southern California house prices using tabular housing data, property images, exploratory data analysis, and an interactive Streamlit dashboard.

## Overview

This project explores how house prices relate to features such as city, street, square footage, bedrooms, bathrooms, and property images. It includes a Kaggle notebook for analysis and modeling, plus a Streamlit deployment for interactive exploration and price prediction.

## Features

- Exploratory data analysis for housing trends
- City, street, square footage, bedroom, and bathroom price insights
- Random Forest tabular baseline model
- Image and tabular deep learning model design using EfficientNet
- Interactive Streamlit dashboard
- Dark HeroUI-inspired interface
- Property image gallery
- House price prediction tool

## Project Structure

```text
socal2/
├── app.py
├── requirements.txt
├── README.md
├── README_STREAMLIT.md
├── socal2.csv
├── socal2 (1).ipynb
├── .streamlit/
│   └── config.toml
└── socal2/
    └── socal_pics/
```

## Dataset

The project uses a Southern California house prices and images dataset. The tabular file contains house price, location, size, bedroom count, bathroom count, and image identifiers. The image folder contains property images linked to the tabular records.

## Exploratory Data Analysis

The notebook answers key analysis questions, including:

- Number of houses in each city
- Top cities by total house prices
- Streets with the largest number of houses
- Average house price in each city
- Relationship between square footage and price
- Relationship between bedrooms and price
- Relationship between bathrooms and price

## Modeling Approach

### Tabular Model

The main deployed model is a Random Forest Regressor trained on structured housing features:

- City encoding
- Bedrooms
- Bathrooms
- Square footage
- Total rooms
- Square feet per room

The model is evaluated using R2 score and Mean Absolute Error.

### Image + Tabular Model

The notebook also includes a deep learning approach that combines image and tabular data:

- EfficientNet extracts image features from property photos.
- Dense layers process tabular housing features.
- Both branches are concatenated before the final regression output.
- The model predicts scaled log house prices for more stable training.

This approach is useful because images can capture information that is not present in the CSV, such as exterior quality, condition, architecture, curb appeal, and visual style.

## Streamlit Dashboard

The Streamlit app provides:

- Dataset overview
- EDA visualizations
- Price prediction interface
- Property image gallery
- Image and tabular model design explanation
- Dark HeroUI-inspired styling

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Open the local app:

```text
http://localhost:8501
```

## Deployment

The app can be deployed on:

- Streamlit Community Cloud
- Hugging Face Spaces
- Render
- Railway

For free deployment, Streamlit Community Cloud is recommended. If the full image folder is too large for the hosting platform, deploy with a smaller image sample or use only the tabular dashboard.

## Key Insights

- Square footage has a strong positive relationship with house price.
- Bathrooms are useful for predicting price and often reflect property size or luxury level.
- Bedrooms have a positive but weaker relationship with price.
- City and location are major pricing factors.
- Images can provide extra visual signals, but tabular features remain essential.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- EfficientNet
- Streamlit
- Plotly
- Pillow

## Author

Created as a machine learning and deep learning project for Southern California house price prediction using tabular and image data.
