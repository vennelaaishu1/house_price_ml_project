# 🏠 Ames House Price Prediction

An end-to-end Machine Learning project that predicts house prices using the **Ames Housing Dataset**.

The project covers the complete Machine Learning lifecycle:

**Data Understanding → EDA → Data Cleaning → Feature Engineering → Preprocessing → Model Training → Model Comparison → Best Model Selection → Model Saving → FastAPI → Streamlit → Deployment**

---

## 🚀 Live Demo

### 🌐 Streamlit Application

👉 https://housepricemlproject-byjqruw2rau89fd5vzn2hu.streamlit.app

The Streamlit application provides a simple user interface where users can enter house details and receive a predicted house price.

### ⚡ FastAPI Backend

👉 https://house-price-ml-project-h58v.onrender.com

### 📚 FastAPI Swagger Documentation

👉 https://house-price-ml-project-h58v.onrender.com/docs

---

# 📌 Project Overview

House prices depend on several factors such as:

- Overall house quality
- Living area
- Basement area
- Garage capacity
- First-floor area
- Year built
- Year remodeled
- Sale year
- Neighborhood
- House features

This project uses historical Ames Housing data to train multiple regression models and select the best-performing model for house price prediction.

---

# 🎯 Objectives

The main objectives of this project are:

1. Understand the Ames Housing dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Analyze missing values.
4. Identify potential outliers.
5. Analyze feature correlations.
6. Clean the dataset.
7. Perform feature engineering.
8. Preprocess numerical and categorical features.
9. Train multiple regression models.
10. Compare model performance.
11. Select the best model.
12. Save the trained model.
13. Build a FastAPI prediction API.
14. Build a Streamlit frontend.
15. Deploy the application online.
16. Generate predictions on the separate test dataset.

---

# 📂 Project Structure

```text
house_price_ml_project/
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── model/
│   └── house_price_model.pkl
│
├── notebooks/
│   ├── house_price_eda.ipynb
│   └── house_price_eda_test.ipynb
│
├── backend/
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── predictions.csv
├── requirements.txt
├── .python-version
├── .gitignore
└── README.md