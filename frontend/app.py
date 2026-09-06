import streamlit as st
import requests

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Ames House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏠 Ames House Price Prediction")
st.write("Predict house prices using a trained Gradient Boosting Regression model.")

st.divider()

# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.number_input(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=7,
        step=1
    )

    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=2000,
        step=50
    )

    garage_cars = st.number_input(
        "Garage Capacity (cars)",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=1000,
        step=50
    )

with col2:
    first_flr_sf = st.number_input(
        "First Floor Area (sq ft)",
        min_value=100,
        max_value=5000,
        value=1200,
        step=50
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2005,
        step=1
    )

    year_remod = st.number_input(
        "Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2010,
        step=1
    )

    year_sold = st.number_input(
        "Year Sold",
        min_value=2006,
        max_value=2026,
        value=2026,
        step=1
    )

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

st.divider()

if st.button("🔮 Predict House Price", use_container_width=True):

    # Data sent to FastAPI
    input_data = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FirstFlrSF": first_flr_sf,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "YrSold": year_sold
    }

    # FastAPI URL
    api_url = "https://house-price-ml-project-h58v.onrender.com/predict"
    
    try:

        # Send request to FastAPI
        response = requests.post(
            api_url,
            json=input_data
        )

        # Check response
        if response.status_code == 200:

            result = response.json()

            predicted_price = result["predicted_price"]

            st.success("Prediction generated successfully!")

            st.metric(
                label="Predicted House Price",
                value=f"${predicted_price:,.2f}"
            )

        else:

            st.error(
                f"API Error: {response.status_code}\n\n"
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FastAPI.\n\n"
            "Make sure the FastAPI server is running on "
            "http://127.0.0.1:8000"
        )

    except Exception as e:

        st.error(f"Unexpected error: {e}")