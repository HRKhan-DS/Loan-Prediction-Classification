import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open("RandomForestModel.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

# Mapping for categorical variables
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
education_map = {"Graduate": 1, "Not Graduate": 0}
self_employed_map = {"Yes": 1, "No": 0}
credit_history_map = {"Yes": 1, "No": 0}
property_area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}

def main():
    st.title("Loan Prediction Classification")

    # Create input fields for user interaction
    st.sidebar.header('Enter Property Details')
    gender = st.sidebar.selectbox("Select gender:", ("Male", "Female"))
    married = st.sidebar.selectbox("Are you married?", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Number of dependents:", (0, 1, 2, 3))
    education = st.sidebar.selectbox("Education:", ("Graduate", "Not Graduate"))
    self_employed = st.sidebar.selectbox("Self Employed:", ("Yes", "No"))
    credit_history = st.sidebar.selectbox("Credit History:", ("Yes", "No"))
    property_area = st.sidebar.selectbox("Select property area:", ("Rural", "Semiurban", "Urban"))
    loan_amount = st.sidebar.number_input("Loan Amount:", min_value=50000)
    loan_amount_term = st.sidebar.number_input("Loan Amount Term:", min_value=8)
    total_income = st.sidebar.number_input("Total Income:", min_value=15000)

    if st.button("Classify"):
        loan_amount_log = np.log(loan_amount)
        loan_amount_term_box = loan_amount_term  # Since no transformation is applied
        total_income_log = np.log(total_income)

        # Convert user input to numerical values
        gender = gender_map[gender]
        married = married_map[married]
        education = education_map[education]
        self_employed = self_employed_map[self_employed]
        credit_history = credit_history_map[credit_history]
        property_area = property_area_map[property_area]

        # Create a dictionary with the user input
        user_input = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'Credit_History': credit_history,
            'Property_Area': property_area,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Total_Income_Log': total_income_log,
            'LoanAmountLog': loan_amount_log,
            'Loan_Amount_Term_Box': loan_amount_term_box,
            'Semiurban': 0,  # You need to set the values based on user input
            'Urban': 1  # You need to set the values based on user input
        }

        # Create a data frame from the user input
        user_input_df = pd.DataFrame([user_input])

        # Make predictions using the trained model (rf_model)
        prediction = rf_model.predict(user_input_df)

        # Display the result
        if prediction[0] == 1:
            st.write("Congratulations! You are eligible for a Loan.")
        else:
            st.write("Sorry, you are not eligible for a Loan")

    # Larger gap using multiple <br> tags
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # Instructions for Loan Eligible Classifier
    st.write("Welcome to the Loan Eligible Classifier!")
    st.write("To determine whether you are eligible for a Home Loan.")
    st.write("Click the 'Classify' button, and the result will be displayed above.")

if __name__ == '__main__':
    main()
