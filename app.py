import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load and preprocess data
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
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f4f9;
        margin: 0;
        padding: 0;
    }
    .app-container {
        max-width: 900px;
        margin: auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    .form-section {
        margin: 20px 0;
    }
    label, h3 {
        color: #34495e;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .result-message {
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .feature-importance-container {
        margin-top: 20px;
        text-align: center;
    }
    /* Responsive design */
    @media (max-width: 768px) {
        .app-container {
            padding: 10px;
        }
        h1 {
            font-size: 2rem;
        }
        .form-section {
            margin: 10px 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# App layout and interactivity
st.markdown("<div class='app-container'>", unsafe_allow_html=True)

st.title(" Diabetes Prediction App")
st.markdown("""
    Welcome to the **Diabetes Prediction App**.  
    Provide your health details below to predict the risk of diabetes.
""")

with st.form("prediction_form"):
    st.markdown("<h3>Enter Your Health Details</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=25, help="Your current age in years.")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Select your gender.")
        hypertension = st.radio("Hypertension", [0, 1], help="0: No, 1: Yes")
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, help="Your Body Mass Index.")

    with col2:
        smoking_history = st.selectbox(
            "Smoking History",
            ["Never", "No Info", "Current", "Former", "Ever", "Not Current"],
            help="Your smoking habits history."
        )
        heart_disease = st.radio("Heart Disease", [0, 1], help="0: No, 1: Yes")
        hba1c = st.number_input("HbA1c Level", min_value=4.0, max_value=15.0, value=5.5, step=0.1, help="Your average blood glucose level over 3 months.")
        blood_glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=250, value=100, help="Your current blood glucose level.")

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    gender_map = {"Male": 1, "Female": 2, "Other": 3}
    smoking_map = {"Never": 1, "No Info": 2, "Current": 3, "Former": 4, "Ever": 5, "Not Current": 6}
    input_data = np.array([[age, gender_map[gender], smoking_map[smoking_history], hypertension, heart_disease, bmi, hba1c, blood_glucose]])

    prediction = rf_model.predict(input_data)
    result = "High risk of diabetes." if prediction[0] == 1 else "Low risk of diabetes."
    color = "red" if prediction[0] == 1 else "green"

    st.markdown(f"<p class='result-message' style='color: {color};'>{result}</p>", unsafe_allow_html=True)

st.markdown("<div class='feature-importance-container'>", unsafe_allow_html=True)
st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = x.columns

plt.figure(figsize=(8, 6))
plt.title("Feature Importance", fontsize=16)
plt.bar(range(x.shape[1]), importances[indices], align="center", color="#3498db")
plt.xticks(range(x.shape[1]), features[indices], rotation=45, ha='right', fontsize=10)
plt.tight_layout()

st.pyplot(plt)

st.markdown("</div></div>", unsafe_allow_html=True)
