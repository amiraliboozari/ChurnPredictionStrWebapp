#We notice that when we look at Info of the dataframe, there is a column called Internet Services that only has 703 non null obvjects 
#and not 1000 like the rest of the columns. So we need to fix that. I present the n/a values appearant with the .isna function

#When we analyze the InternetServices column we notice that there are common strings values upon the column and with this in mind I decided to use the .fillna command to fill the n/as with an empty space string
#and replaced the original df by creating the same df but without the n/a values in InternetServices column.
# The describe function shows min and max in all columns and so explain what these findings are and how they correlate with the business tasks and data
# We find that the correlation and find which apsects of the data have a positive correlation with two or more columns at once
# Gender -> 1 Female  0 Male
# Age -> 10 to 100
# Tenure -> 0 to 130
# MonthlyCharges -> 0.0 to 150.0
# Churn -> 1 Yes 0 No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# The model is trained and saved as model.pkl
# Order of the X  -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

#Load the scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction Streamlit App")
st.write("This is a simple app to predict customer churn using a machine learning model.")

st.divider()

st.write("Please enter the values and press the predection button for our prediction.")

st.divider()

age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

monthly_charges = st.number_input("Enter Monthly Charges", min_value=0.0, max_value=150.0)

gender = st.selectbox("Enter the Gender", ["Male","Female"])

st.divider()

predictionbutton = st.button("Predict")

st.divider()

if predictionbutton:

    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, tenure, monthly_charges]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.balloons()

    st.write(f"Predicted: {predicted}")


else: 
    st.write("Please enter the values and press the predection button for our prediction.")