import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    package = joblib.load("house_price_model.pkl")
    return package["pipeline"]


loaded_pipeline = load_model()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_house(raw_house):

    house_df = pd.DataFrame([raw_house])

    # Dataset contains sales from 2016
    reference_year = 2016

    # ------------------------------
    # Feature engineering
    # ------------------------------

    house_df["Month"] = house_df["Sale Month"]

    house_df["renovated"] = (
        house_df["Renovation Year"] != 0
    ).astype(int)

    house_df["House Age"] = (
        reference_year - house_df["Built Year"]
    )

    house_df["year since renovation"] = np.where(
        house_df["Renovation Year"] == 0,
        0,
        reference_year - house_df["Renovation Year"]
    )

    # Remove helper column
    house_df.drop(columns=["Sale Month"], inplace=True)

    # ------------------------------
    # Prediction
    # ------------------------------

    predicted_log_price = loaded_pipeline.predict(house_df)

    # Convert log(price) back to price
    predicted_price = np.expm1(predicted_log_price[0])

    return predicted_price


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

st.title("🏠 House Price Prediction")

st.write(
    "Enter the house details below to estimate its selling price."
)


# --------------------------------------------------
# Input fields
# --------------------------------------------------

number_of_bedrooms = st.number_input(
    "Number of bedrooms",
    min_value=0,
    max_value=20,
    value=3,
    step=1
)

number_of_bathrooms = st.number_input(
    "Number of bathrooms",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.5
)

living_area = st.number_input(
    "Living area",
    min_value=0,
    value=1800,
    step=1
)

lot_area = st.number_input(
    "Lot area",
    min_value=0,
    value=5000,
    step=1
)

number_of_floors = st.number_input(
    "Number of floors",
    min_value=1.0,
    max_value=10.0,
    value=1.0,
    step=0.5
)

waterfront_present = st.selectbox(
    "Waterfront present",
    [0, 1]
)

number_of_views = st.selectbox(
    "Number of views",
    [0, 1, 2, 3, 4]
)

condition_of_house = st.selectbox(
    "Condition of the house",
    [1, 2, 3, 4, 5]
)

grade_of_house = st.selectbox(
    "Grade of the house",
    list(range(4, 14))
)

area_excluding_basement = st.number_input(
    "Area of the house (excluding basement)",
    min_value=0,
    value=1400,
    step=1
)

area_of_basement = st.number_input(
    "Area of the basement",
    min_value=0,
    value=400,
    step=1
)

built_year = st.number_input(
    "Built Year",
    min_value=1900,
    max_value=2016,
    value=1995,
    step=1
)

renovation_year = st.number_input(
    "Renovation Year (0 if never renovated)",
    min_value=0,
    max_value=2016,
    value=0,
    step=1
)

postal_code = st.number_input(
    "Postal Code",
    min_value=0,
    value=122048,
    step=1
)

latitude = st.number_input(
    "Latitude",
    value=52.2
)

longitude = st.number_input(
    "Longitude",
    value=-114.2
)

living_area_renov = st.number_input(
    "Living area after renovation",
    min_value=0,
    value=1400,
    step=1
)

lot_area_renov = st.number_input(
    "Lot area after renovation",
    min_value=0,
    value=5000,
    step=1
)

schools_nearby = st.number_input(
    "Number of schools nearby",
    min_value=0,
    value=1,
    step=1
)

distance_from_airport = st.number_input(
    "Distance from airport",
    min_value=0,
    value=50,
    step=1
)

sale_month = st.selectbox(
    "Sale Month",
    list(range(5, 13)),
    index=3
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Price"):

    new_house = {
        "number of bedrooms": number_of_bedrooms,
        "number of bathrooms": number_of_bathrooms,
        "living area": living_area,
        "lot area": lot_area,
        "number of floors": number_of_floors,
        "waterfront present": waterfront_present,
        "number of views": number_of_views,
        "condition of the house": condition_of_house,
        "grade of the house": grade_of_house,
        "Area of the house(excluding basement)": area_excluding_basement,
        "Area of the basement": area_of_basement,
        "Built Year": built_year,
        "Renovation Year": renovation_year,
        "Postal Code": postal_code,
        "Lattitude": latitude,
        "Longitude": longitude,
        "living_area_renov": living_area_renov,
        "lot_area_renov": lot_area_renov,
        "Number of schools nearby": schools_nearby,
        "Distance from the airport": distance_from_airport,
        "Sale Month": sale_month
    }

    try:
        predicted_price = predict_house(new_house)

        st.success(
            f"Estimated House Price: ₹{predicted_price:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")