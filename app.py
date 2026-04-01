import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Diabetes Predictor", layout="wide")

st.title("🩺 Diabetes Predictor")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Exploratory Data Analysis", "Model Training", "Make a Prediction"])

@st.cache_data
def load_data():
    return pd.read_csv('diabetes.csv')

if page == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    try:
        df = load_data()
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        st.subheader("Statistical Summary")
        st.write(df.describe())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Class Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            df['Outcome'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'], ax=ax)
            ax.set_xticklabels(['No Diabetes', 'Diabetes'], rotation=0)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Glucose Level Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(df['Glucose'], kde=True, color='steelblue', ax=ax)
            st.pyplot(fig)
            
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.error("Error: 'diabetes.csv' not found. Please ensure the dataset is in the same folder.")

elif page == "Model Training":
    st.header("⚙️ Model Training & Evaluation")
    st.write("Click the button below to preprocess the data, train the Logistic Regression model, and save it.")
    
    if st.button("Train Model Now"):
        with st.spinner("Training in progress..."):
            df = load_data()
            
            cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
            df.fillna(df.median(), inplace=True)
            
            X = df.drop('Outcome', axis=1)
            y = df['Outcome']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            
            joblib.dump(model, 'logistic_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            
            st.success(f"Model trained successfully! Accuracy: **{acc * 100:.2f}%**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['No Diabetes', 'Diabetes'],
                            yticklabels=['No Diabetes', 'Diabetes'], ax=ax)
                st.pyplot(fig)

elif page == "Make a Prediction":
    st.header("🔮 Patient Prediction Form")
    
    if not os.path.exists('logistic_model.pkl') or not os.path.exists('scaler.pkl'):
        st.warning("Model files not found! Please go to the 'Model Training' tab and train the model first.")
    else:
        st.write("Enter the patient's medical details below to predict their risk of diabetes.")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input("Pregnancies (count)", min_value=0, max_value=20, value=3)
                glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=117.0)
                blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=72.0)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=29.0)
                
            with col2:
                insulin = st.number_input("Insulin (μU/mL)", min_value=0.0, max_value=1000.0, value=125.0)
                bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=32.3)
                dpf = st.number_input("Diabetes Pedigree Function (score)", min_value=0.000, max_value=3.000, value=0.370, format="%.3f")
                age = st.number_input("Age (years)", min_value=0, max_value=120, value=29)

            submit_button = st.form_submit_button(label="Predict Outcome")
            
        if submit_button:
            model = joblib.load('logistic_model.pkl')
            scaler = joblib.load('scaler.pkl')
            
            user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            scaled_data = scaler.transform(user_data)
            
            prediction = model.predict(scaled_data)
            probabilities = model.predict_proba(scaled_data)[0]
            
            st.markdown("---")
            if prediction[0] == 1:
                st.error(f"### Result: High Risk of Diabetes")
                st.write(f"Confidence: {probabilities[1] * 100:.1f}%")
            else:
                st.success(f"### Result: Low Risk (No Diabetes)")
                st.write(f"Confidence: {probabilities[0] * 100:.1f}%")
