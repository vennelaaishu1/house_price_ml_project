# Ames House Price Prediction

Ready-to-open VS Code project structure.

## Project flow
Jupyter EDA → Data Cleaning → Feature Engineering → Preprocessing
→ 4 Regression Models → Model Comparison → Best Model
→ FastAPI → Streamlit → Deployment

## Important
The trained binary model must be copied from the Jupyter-generated `model/house_price_model.pkl`
into this project's `model/` folder.

The Ames `train.csv` used for training should be copied into `dataset/`.

## Current phase
Jupyter ML work is complete. The next implementation step is FastAPI in `backend/main.py`.
