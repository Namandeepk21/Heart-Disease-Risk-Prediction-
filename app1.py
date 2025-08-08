import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="💓 Heart Disease Predictor", page_icon="💖", layout="centered")

# Load model, scaler, expected columns
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
            padding: 20px;
            border-radius: 15px;
        }
        h1 {
            color: #d63384;
            text-align: center;
            font-size: 42px;
            font-family: 'Arial Rounded MT Bold';
        }
        .stButton>button {
            background-color: #d63384;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .stSlider, .stSelectbox, .stNumberInput {
            padding: 5px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>💓 Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("### 🧾 Fill out your health details:")

# Input form
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("🎂 Age", 18, 100, 40)
        sex = st.selectbox("⚧️ Sex", ["M", "F"])
        chest_pain = st.selectbox("💢 Chest Pain Type", ["ATA(Mild chest discomfort, may not be heart-related)", "NAP(Chest pain not related to heart problems)", "TA(Classic chest pain due to blocked arteries)", "ASY(No chest pain, but may still have heart issues)"])
        resting_bp = st.number_input("💉 Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("🥩 Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

    with col2:
        resting_ecg = st.selectbox("Resting ECG Result", ["Normal(Normal heart rhythm)", "ST(May indicate minor heart stress or earlier heart issues)", "LVH(Thick heart muscle wall – a sign of chronic pressure or heart strain)"])
        max_hr = st.slider("🏃 Max Heart Rate During Exercise", 60, 220, 150)
        exercise_angina = st.selectbox("🏋️ Angina During Exercise (Chest Pain)", ["Y", "N"])
        oldpeak = st.slider("📉 ECG Depression Level (Oldpeak)", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("📈 Heart Stress Graph Slope (ST Slope)", ["Up", "Flat", "Down"])

st.markdown("---")

# Predict button
if st.button("🔍 Predict Now"):
    with st.spinner("Analyzing your health data... ⏳"):
        # Prepare input
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([raw_input])

        # Fill missing columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        st.markdown("### 🧾 Your Input Summary:")
        st.dataframe(input_df)

        # Result with emoji and message
        if prediction == 1:
            st.error("😟 **High Risk of Heart Disease Detected!**\n\nPlease consult a heart specialist.")
            st.markdown("💔 Take care of your health. Eat well. Get tested.")
        else:
            st.success("😊 **Low Risk of Heart Disease!**\n\nYou seem to be in a safe range.")
            st.markdown("💖 Keep maintaining a healthy lifestyle!")


