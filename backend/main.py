from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os


# =========================================================
# 1. CREATE FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Ames House Price Prediction API",
    description="House price prediction using Gradient Boosting",
    version="1.0"
)


# =========================================================
# 2. PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "house_price_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "train.csv"
)


# =========================================================
# 3. LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# 4. LOAD TRAINING DATA
# =========================================================

train_data = pd.read_csv(DATA_PATH)


# =========================================================
# 5. CREATE DEFAULT VALUES
# =========================================================

# Remove target and ID
default_data = train_data.drop(
    columns=["SalePrice", "Id"]
).copy()


# =========================================================
# 6. SAME FEATURE ENGINEERING USED IN JUPYTER
# =========================================================

default_data["TotalSF"] = (
    default_data["TotalBsmtSF"].fillna(0)
    + default_data["1stFlrSF"].fillna(0)
    + default_data["2ndFlrSF"].fillna(0)
)

default_data["TotalBathrooms"] = (
    default_data["FullBath"].fillna(0)
    + 0.5 * default_data["HalfBath"].fillna(0)
    + default_data["BsmtFullBath"].fillna(0)
    + 0.5 * default_data["BsmtHalfBath"].fillna(0)
)

default_data["TotalPorchSF"] = (
    default_data["OpenPorchSF"].fillna(0)
    + default_data["3SsnPorch"].fillna(0)
    + default_data["EnclosedPorch"].fillna(0)
    + default_data["ScreenPorch"].fillna(0)
    + default_data["WoodDeckSF"].fillna(0)
)

default_data["HouseAge"] = (
    default_data["YrSold"]
    - default_data["YearBuilt"]
)

default_data["YearsSinceRemodel"] = (
    default_data["YrSold"]
    - default_data["YearRemodAdd"]
)


# =========================================================
# 7. CREATE DEFAULT HOUSE
# =========================================================

# Median for numeric columns
numeric_defaults = default_data.select_dtypes(
    include="number"
).median()


# Mode for categorical columns
categorical_defaults = default_data.select_dtypes(
    exclude="number"
).mode().iloc[0]


default_values = {}

default_values.update(
    numeric_defaults.to_dict()
)

default_values.update(
    categorical_defaults.to_dict()
)


# =========================================================
# 8. INPUT DATA MODEL
# =========================================================

class HouseData(BaseModel):

    # Main user inputs

    OverallQual: int = 5

    GrLivArea: float = 1500

    GarageCars: float = 2

    TotalBsmtSF: float = 1000

    FirstFlrSF: float = 1000

    YearBuilt: int = 2000

    YearRemodAdd: int = 2000

    YrSold: int = 2026


# =========================================================
# 9. HOME ENDPOINT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Ames House Price Prediction API is running!",
        "model": "Gradient Boosting Regressor",
        "status": "success"
    }


# =========================================================
# 10. PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict_price(house: HouseData):

    # Start with default values for all 84 features
    input_values = default_values.copy()


    # Override important user-provided values
    input_values["OverallQual"] = house.OverallQual
    input_values["GrLivArea"] = house.GrLivArea
    input_values["GarageCars"] = house.GarageCars
    input_values["TotalBsmtSF"] = house.TotalBsmtSF
    input_values["1stFlrSF"] = house.FirstFlrSF
    input_values["YearBuilt"] = house.YearBuilt
    input_values["YearRemodAdd"] = house.YearRemodAdd
    input_values["YrSold"] = house.YrSold


    # Recalculate engineered features
    input_values["TotalSF"] = (
        input_values["TotalBsmtSF"]
        + input_values["1stFlrSF"]
        + input_values["2ndFlrSF"]
    )

    input_values["HouseAge"] = (
        input_values["YrSold"]
        - input_values["YearBuilt"]
    )

    input_values["YearsSinceRemodel"] = (
        input_values["YrSold"]
        - input_values["YearRemodAdd"]
    )


    # Create DataFrame
    input_data = pd.DataFrame([input_values])


    # Make prediction
    prediction = model.predict(input_data)


    # Return result
    return {
        "predicted_price": round(
            float(prediction[0]),
            2
        )
    }