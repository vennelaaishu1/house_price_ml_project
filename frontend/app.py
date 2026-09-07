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

st.title("🏠 Ames House Price Prediction")
st.write(
    "Predict house prices using a trained Gradient Boosting Regression model."
)

# --------------------------------------------------
# API URL
# --------------------------------------------------

API_URL = "https://house-price-ml-project-h58v.onrender.com"

# --------------------------------------------------
# Single House Prediction
# --------------------------------------------------

st.divider()
st.header("🏠 Single House Prediction")

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

st.divider()

if st.button(
    "🔮 Predict House Price",
    use_container_width=True
):

    input_data = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FirstFlrSF": first_flr_sf,
        "YearRemodAdd": year_remod,
        "YearBuilt": year_built,
        "YrSold": year_sold
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=input_data,
            timeout=60
        )

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
            "❌ Could not connect to FastAPI."
        )

    except requests.exceptions.Timeout:

        st.error(
            "⏳ Request timed out. Please try again."
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )


# --------------------------------------------------
# Batch Prediction
# --------------------------------------------------

st.divider()
st.header("📂 Batch House Price Prediction")

st.write(
    "Upload a CSV file containing multiple houses. "
    "The API will generate predictions for all rows."
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    st.success(
        f"File uploaded: {uploaded_file.name}"
    )

    if st.button(
        "🚀 Predict All Houses",
        use_container_width=True
    ):

        with st.spinner(
            "Generating predictions for all houses..."
        ):

            try:

                file_bytes = uploaded_file.getvalue()

                files = {
                    "file": (
                        uploaded_file.name,
                        file_bytes,
                        "text/csv"
                    )
                }

                response = requests.post(
                    f"{API_URL}/predict-batch",
                    files=files,
                    timeout=120
                )

                if response.status_code == 200:

                    st.success(
                        "✅ Predictions generated successfully!"
                    )

                    # Download button
                    st.download_button(
                        label="⬇️ Download predictions.csv",
                        data=response.content,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    # Show preview
                    try:

                        import pandas as pd
                        import io

                        prediction_df = pd.read_csv(
                            io.BytesIO(response.content)
                        )

                        st.subheader(
                            "📊 Prediction Preview"
                        )

                        st.dataframe(
                            prediction_df.head(10),
                            use_container_width=True
                        )

                        st.write(
                            f"Total predictions: "
                            f"**{len(prediction_df)}**"
                        )

                    except Exception as e:

                        st.warning(
                            f"Could not display preview: {e}"
                        )

                else:

                    st.error(
                        f"Batch API Error: "
                        f"{response.status_code}\n\n"
                        f"{response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to FastAPI."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏳ Batch prediction timed out. "
                    "Please try again."
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Ames House Price Prediction | "
    "Gradient Boosting Regression"
)