import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

data = pd.read_csv(r"./diabetes_prediction_dataset.csv")
data["gender"] = data["gender"].map({"Male": 1, "Female": 2, "Other": 3})
data["smoking_history"] = data["smoking_history"].map({
    "never": 1, "No Info": 2, "current": 3, "former": 4, "ever": 5, "not current": 6
})

y = data['diabetes']
x = data.drop("diabetes", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)


# Apply custom CSS styling
st.markdown("""
    <style>
   /* styles.css */
body {
    font-family: 'Arial', sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f7f9fc;
    color: #333;
}

header {
    background-color: #4CAF50;
    color: white;
    padding: 20px 0;
    text-align: center;
}

.header-container h1 {
    margin: 0;
    font-size: 2.5rem;
}

.header-container p {
    font-size: 1.2rem;
    margin-top: 10px;
}

main {
    padding: 20px;
}

.form-section {
    max-width: 600px;
    margin: 0 auto;
    background-color: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.prediction-form .form-group {
    margin-bottom: 15px;
}

.prediction-form label {
    display: block;
    font-weight: bold;
    margin-bottom: 5px;
}

.prediction-form input,
.prediction-form select {
    width: 100%;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
}

.prediction-form .submit-button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    padding: 10px;
    font-size: 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.prediction-form .submit-button:hover {
    background-color: #45a049;
}

.footer-container {
            text-align: center;
            margin-top: 50px;
            padding: 10px;
            background-color: #6200ea;
            color: white;
            border-radius: 8px;
}
.footer-container p {
            margin: 0;
            font-size: 0.9rem;
}

    </style>
    """, unsafe_allow_html=True)

# App layout and interactivity
st.markdown(
    """
    <div class="header-container">
        <h1>Diabetes Prediction</h1>
        <p>Accurately Predict Diabetes Risk with Advanced Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    st.markdown("### Enter Your Health Details")
    
    gender = st.selectbox(
        "Gender", 
        ["Select Gender", "Male", "Female", "Other"], 
        index=0, 
        help="Select your gender."
    )
    age = st.number_input(
        "Age", 
        min_value=1, 
        max_value=120, 
        value=25, 
        step=1, 
        help="Enter your age in years."
    )
    hypertension = st.selectbox(
        "Hypertension", 
        ["No", "Yes"], 
        index=0, 
        help="Do you have hypertension?"
    )
    heart_disease = st.selectbox(
        "Heart Disease", 
        ["No", "Yes"], 
        index=0, 
        help="Do you have a heart disease history?"
    )
    smoking_history = st.selectbox(
        "Smoking History", 
        ["Select an Option", "Never", "No Info", "Current", "Former", "Ever", "Not Current"], 
        index=0, 
        help="Select your smoking history."
    )
    bmi = st.number_input(
        "BMI", 
        min_value=10.0, 
        max_value=50.0, 
        value=25.0, 
        step=0.1, 
        help="Enter your Body Mass Index."
    )
    hba1c = st.number_input(
        "HbA1c Level", 
        min_value=4.0, 
        max_value=15.0, 
        value=5.5, 
        step=0.1, 
        help="Enter your HbA1c Level (average blood glucose level over 3 months)."
    )
    blood_glucose = st.number_input(
        "Blood Glucose Level", 
        min_value=50, 
        max_value=250, 
        value=100, 
        step=1, 
        help="Enter your current Blood Glucose Level."
    )
    
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # Map inputs to numeric values
    gender_map = {"Male": 1, "Female": 2, "Other": 3}
    smoking_map = {
        "Never": 1, "No Info": 2, "Current": 3, 
        "Former": 4, "Ever": 5, "Not Current": 6
    }
    
    # Check if a valid gender and smoking history option are selected
    if gender == "Select Gender" or smoking_history == "Select an Option":
        st.error("Please select valid options for Gender and Smoking History.")
    else:
        # Convert categorical inputs to numeric
        gender_value = gender_map[gender]
        smoking_value = smoking_map[smoking_history]
        hypertension_value = 1 if hypertension == "Yes" else 0
        heart_disease_value = 1 if heart_disease == "Yes" else 0

        # Create input array for prediction
        input_data = pd.DataFrame(
    [[age, gender_value, hypertension_value, heart_disease_value, smoking_value, bmi, hba1c, blood_glucose]],
    columns=x.columns )
        
        # Placeholder for prediction model (replace this with actual model logic)
        prediction = rf_model.predict(input_data)  # Dummy prediction for demonstration
        result = "Diabetic." if prediction == 1 else "Not Diabetic."
        color = "red" if prediction == 1 else "green"

        st.markdown(
            f"<p style='text-align:center; color:{color}; font-size:20px;'>{result}</p>", 
            unsafe_allow_html=True
        )
st.markdown(
    """
    <div class="footer-container">
        <p>© 2024 Diabetes Prediction App. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True,
)        
