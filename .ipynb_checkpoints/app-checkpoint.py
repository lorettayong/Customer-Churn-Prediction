# Project: Telco Customer Churn Prediction
# Phase 8: Model Deployment
# Step 2: Streamlit Application (app.py)

import streamlit as st
import pandas as pd
import joblib

# Title
st.title("Telco Customer Churn Prediction")

# Load the full pipeline
MODEL_PIPELINE_PATH = 'model_pipeline.joblib'

try:
    model_pipeline = joblib.load(MODEL_PIPELINE_PATH)
    st.success("Model pipeline loaded successfully.")
except FileNotFoundError:
    st.error(f"Error: File '{MODEL_PIPELINE_PATH}' not found. Please ensure the pipeline is in the same directory as app.py.")
    st.stop()

# Function to collect user input
def user_input_features():
    st.subheader("Enter Customer Information:")

    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen_label = st.selectbox('Senior Citizen', ['No', 'Yes'])
    senior_citizen = 1 if senior_citizen_label == 'Yes' else 0
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No internet service', 'No', 'Yes'])
    online_backup = st.selectbox('Online Backup', ['No internet service', 'No', 'Yes'])
    device_protection = st.selectbox('Device Protection', ['No internet service', 'No', 'Yes'])
    tech_support = st.selectbox('Tech Support', ['No internet service', 'No', 'Yes'])
    streaming_tv = st.selectbox('Streaming TV', ['No internet service', 'No', 'Yes'])
    streaming_movies = st.selectbox('Streaming Movies', ['No internet service', 'No', 'Yes'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method = st.selectbox('Payment Method', [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0)
    total_charges = st.number_input('Total Charges', min_value=0.0)

    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    return pd.DataFrame([data])

# Get user input
input_df = user_input_features()

# Predict only when the user clicks the button
if st.button("Predict Churn"):
    try:
        prediction = model_pipeline.predict(input_df)[0]
        probability = model_pipeline.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.markdown("### 🔴 **Churn Prediction: Yes**")
            st.markdown("This customer is likely to **CHURN!**")
            st.markdown("Consider proactive retention strategies for this customer.")
        else:
            st.markdown("### 🟢 **Churn Prediction: No**")
            st.markdown("This customer is likely **NOT to churn.**")
            st.markdown("This customer appears stable. Maintain service satistfaction and continue engagement.")

        st.markdown(f"**Churn Probability:** {probability:.2%}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
