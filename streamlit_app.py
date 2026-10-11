
import streamlit as st
import pandas as pd
import requests
from pathlib import Path

st.set_page_config(
    page_title="Bangalore House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Bengaluru_House_Data.csv"
API_URL = "http://127.0.0.1:8000/predict"

st.title("🏠 Bangalore House Price Prediction")
st.markdown(
    "### Estimate property prices using Machine Learning"
)
st.write(
    "Enter the property details below to receive an estimated "
    "price from our trained Random Forest model."
)

# Load location and area-type options from the original dataset
@st.cache_data
def load_options():
    df = pd.read_csv(DATA_PATH)

    locations = sorted(
        df["location"].dropna().astype(str).str.strip().unique()
    )

    area_types = sorted(
        df["area_type"].dropna().astype(str).str.strip().unique()
    )

    return locations, area_types

try:
    locations, area_types = load_options()
except Exception as exc:
    st.error(f"Could not load dataset options: {exc}")
    st.stop()

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("🏡 Property Details")

    with st.form("prediction_form"):
        location = st.selectbox(
            "Location",
            locations,
            index=locations.index("Whitefield")
            if "Whitefield" in locations else 0
        )

        total_sqft = st.number_input(
            "Total area (sq. ft.)",
            min_value=100.0,
            max_value=1000000.0,
            value=1200.0,
            step=50.0
        )

        bhk = st.selectbox(
            "Bedrooms (BHK)",
            list(range(1, 11)),
            index=1
        )

        bath = st.selectbox(
            "Number of bathrooms",
            list(range(1, 11)),
            index=1
        )

        balcony = st.selectbox(
            "Number of balconies",
            list(range(0, 6)),
            index=1
        )

        area_type = st.selectbox(
            "Area type",
            area_types
        )

        submitted = st.form_submit_button(
            "🔍 Predict House Price",
            use_container_width=True
        )

with right:
    st.subheader("📊 Prediction Result")

    st.info(
        "Complete the property details and click "
        "'Predict House Price' to view the estimate."
    )

    if submitted:
        payload = {
            "location": location,
            "total_sqft": float(total_sqft),
            "bhk": int(bhk),
            "bath": float(bath),
            "balcony": float(balcony),
            "area_type": area_type
        }

        try:
            with st.spinner("Predicting price..."):
                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=30
                )

            if response.ok:
                result = response.json()

                price_lakhs = result["predicted_price_lakhs"]
                price_rupees = result["predicted_price_rupees"]

                st.success("Prediction completed!")

                st.metric(
                    label="Estimated House Price",
                    value=f"₹{price_lakhs:,.2f} lakhs"
                )

                st.write(f"Approximately ₹{price_rupees:,.0f}")

                st.caption(
                    f"Location: {location} | "
                    f"Area: {total_sqft:,.0f} sq. ft. | "
                    f"Configuration: {bhk} BHK"
                )

                st.warning(
                    "This is a machine-learning estimate, not a "
                    "guaranteed market price. Actual prices vary "
                    "with property condition, amenities and market conditions."
                )

            else:
                st.error(
                    f"API returned status {response.status_code}: "
                    f"{response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. Make sure the API server "
                "is running on port 8000."
            )

        except requests.exceptions.Timeout:
            st.error("The prediction request timed out. Please retry.")

        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")

st.divider()
st.caption("Bangalore House Price Prediction | Machine Learning Project")
