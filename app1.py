import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.graph_objects as go
import plotly.express as px

# Function to create a download link for a dataframe as a csv file
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a herf="data:file/csv;base64,{b64}" download="Prediction.csv">Download Predictions CSV </a>'
    return href

# -------------------------------------------
# Title and Tabs
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("💖 Heart Disease Predictor")

tab1, tab2, tab3 = st.tabs(['🩺 Predict', '📊 Bulk Predict', '📈 Model Information'])

# -------------------------------------------
# TAB 1: Single Prediction
with tab1:
    st.header("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age (years)", min_value=0, max_value=120)
        Sex = st.selectbox("Sex", ["Male", "Female"])
        ChestPainType = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
        Cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=1000)
        FastingBS = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])

    with col2:
        RestingECG = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        MaxHR = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=220)
        ExerciseAngina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        Oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0)
        ST_Slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

    # Encoding categorical features
    Sex = 0 if Sex == 'Male' else 1
    ChestPainType = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(ChestPainType)
    FastingBS = 1 if FastingBS == "> 120 mg/dl" else 0
    RestingECG = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(RestingECG)
    ExerciseAngina = 1 if ExerciseAngina == "Yes" else 0
    ST_Slope = ["Upsloping", "Flat", "Downsloping"].index(ST_Slope)

    input_data = pd.DataFrame({
        'Age': [Age], 'Sex': [Sex], 'ChestPainType': [ChestPainType],
        'RestingBP': [RestingBP], 'Cholesterol': [Cholesterol], 'FastingBS': [FastingBS],
        'RestingECG': [RestingECG], 'MaxHR': [MaxHR], 'ExerciseAngina': [ExerciseAngina],
        'Oldpeak': [Oldpeak], 'ST_Slope': [ST_Slope]
    })

    algonames = ['Logistic Regression']
    modelnames = ['LogisticRegression.pkl']

    predictions = []

    def predict_heart_disease(data):
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    if st.button("🔮 Predict"):
        st.markdown("---")
        result = predict_heart_disease(input_data)

        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.success("🎉 You are NOT likely to have heart disease.")

                fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=0,
                title={'text': "Risk Level"},
                gauge={'axis': {'range': [0, 1]},
                    'bar': {'color': "green"},
                    'steps': [{'range': [0, 0.5], 'color': "lightgreen"},
                         {'range': [0.5, 1], 'color': "lightcoral"}]}
                    ))
                st.plotly_chart(fig, use_container_width=True)

                # ✅ Add healthy recommendations for no heart disease
                st.subheader("✅ Health Recommendations:")
                st.markdown("""
                    - Maintain a balanced diet 🥗
                    - Regular physical activity 🏃‍♂️
                    - Avoid smoking 🚭
                    - Limit alcohol consumption 🍷
                    - Manage stress levels 🧘‍♂️
                    - Get routine medical checkups 🏥
                    """)

            else:
                st.error("⚠️ You are LIKELY to have heart disease. Please consult your doctor.")

                fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=1,
                title={'text': "Risk Level"},
                gauge={'axis': {'range': [0, 1]},
                    'bar': {'color': "red"},
                    'steps': [{'range': [0, 0.5], 'color': "lightgreen"},
                             {'range': [0.5, 1], 'color': "lightcoral"}]}
                        ))
                st.plotly_chart(fig, use_container_width=True)

            # ✅ Add recommendations for people who may have heart disease
                st.subheader("⚠️ Medical Recommendations:")
                st.markdown("""
                - Schedule appointment with cardiologist 🩺
                - Follow prescribed medications 💊
                - Adopt heart-healthy diet 🥦
                - Moderate exercise as per doctor's advice 🚶‍♀️
                - Regular monitoring of cholesterol, BP, sugar 🔬
                """)


# -------------------------------------------
# TAB 2: Bulk Prediction
with tab2:
    st.header("Upload CSV for Bulk Prediction")

    st.info("""
        ⚠️ Make sure your CSV file contains following columns:\n  
        1. No NaN values allowed.
        2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR','ExerciseAngina', 'Oldpeak', 'ST_Slope' )\n
        3. Check the spelling of the feature names.
        4. Feature value conventions: \n
            - Age: age of the patient [years]\n
            - Sex: sex of the patient [0: Male, 1: Female]\n
            - ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]\n
            - RestingBP: resting blood pressure [mm Hg]\n
            - Cholesterol: serum cholesterol [mm/dl]\n
            - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]\n
            - RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST  elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]\n
            - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]\n
            - ExerciseAngina: exercise-induced angina [1: Yes, 0: No]\n
            - Oldpeak: oldpeak = ST [Numeric value measured in depression]\n
            - ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]\n
    """)

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        model = pickle.load(open('LogisticRegression.pkl', 'rb'))

        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                             'FastingBS', 'RestingECG', 'MaxHR','ExerciseAngina','Oldpeak','ST_Slope']

        if set(expected_columns).issubset(input_data.columns):
            input_data['Prediction LR'] = model.predict(input_data)

            st.success("✅ Prediction Complete!")
            st.write(input_data)

        else:
            st.warning("Uploaded file has wrong columns!")

    else:
        st.info("Please upload a file to start prediction.")

# -------------------------------------------
# TAB 3: Model Information
with tab3:
    st.header("Model Performance Overview")

    data = {'Decision Tree': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 86.2}
    Models = list(data.keys())
    Accuracies = list(data.values())

    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])

    fig = px.bar(
        df, x='Models', y='Accuracies', text='Accuracies',
        color='Models', color_discrete_sequence=px.colors.qualitative.Set2,
        title="Model Accuracy Comparison"
    )

    fig.update_traces(
        texttemplate='%{text:.2f}%', textposition='outside',
        marker_line_color='black', marker_line_width=1.5
    )

    fig.update_layout(
        yaxis_title='Accuracy (%)',
        xaxis_title='Models',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        hovermode='x unified',
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    st.header("Explaining the Terms used for Heart disease :")
    with st.expander("\U0001F4DD Learn about Chest Pain Types"):
        st.markdown("""
        **Typical Angina (TA):** 
        - Classic chest pain associated with heart problems. 
        - Described as pressure, tightness, or squeezing in the chest. 
        - Triggered by exertion or stress, relieved by rest or medication.
        
        **Atypical Angina (ATA):** 
        - Chest pain doesn't match typical pattern. 
        - May feel sharp or stabbing, located differently. 
        - May not be relieved by rest.
        
        **Non-Anginal Pain (NAP):** 
        - Chest pain not caused by heart problems. 
        - Can be due to digestive issues, musculoskeletal pain, or anxiety.
        
        **Asymptomatic (ASY):** 
        - No chest pain or other symptoms even with heart problems. 
        - Silent ischemia may occur (reduced blood flow without pain).
        """)
    
    with st.expander("🩸 Learn about Resting Blood Pressure (RestingBP)"):
        st.markdown("""
    **Resting Blood Pressure (RestingBP):**
    - Measurement of blood pressure when the body is at rest.
    - Expressed as two numbers: systolic (upper) and diastolic (lower), both in mmHg.
    - **Normal Range:** Around **120/80 mmHg** for adults.
    - **High Blood Pressure (Hypertension):** Systolic ≥ 130 mmHg or Diastolic ≥ 80 mmHg.
    - High RestingBP can be an indicator of increased heart disease risk.
        """)
    
        # Visualization using Plotly
        fig_bp = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 120,
            title = {'text': "Normal Resting BP"},
            gauge = {
                'axis': {'range': [0, 200]},
                'steps': [
                    {'range': [0, 120], 'color': "lightgreen"},
                    {'range': [120, 140], 'color': "yellow"},
                    {'range': [140, 160], 'color': "orange"},
                    {'range': [160, 200], 'color': "red"}],
                'bar': {'color': "darkgreen"}
                }
            ))
        st.plotly_chart(fig_bp, use_container_width=True)

        
    with st.expander("🥚 Learn about Cholesterol"):
        st.markdown("""
        **Cholesterol:**
        - A fatty substance found in the blood, essential for building healthy cells.
        - High levels can increase the risk of heart disease.
        
        **Types of Cholesterol:**
        - **LDL (Low-Density Lipoprotein):** "Bad" cholesterol; too much leads to buildup in arteries.
        - **HDL (High-Density Lipoprotein):** "Good" cholesterol; helps remove LDL.
        
        **Normal Cholesterol Levels (mg/dL):**
        - Total Cholesterol: **Less than 200**
        - LDL: **Less than 100**
        - HDL: **40 or higher**

        - Higher total cholesterol and LDL levels increase the risk of heart disease.
        """)

        # Visualization using Plotly
        fig_chol = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 180,
            title = {'text': "Total Cholesterol"},
            gauge = {
                'axis': {'range': [0, 300]},
                'steps': [
                    {'range': [0, 200], 'color': "lightgreen"},
                    {'range': [200, 240], 'color': "orange"},
                    {'range': [240, 300], 'color': "red"}],
                'bar': {'color': "darkgreen"}
            }
        ))
        st.plotly_chart(fig_chol, use_container_width=True)

   
    with st.expander("🍬 Learn about Fasting Blood Sugar (FastingBS)"):
        st.markdown("""
    **Fasting Blood Sugar (FastingBS):**
    - Blood sugar level after at least 8 hours of fasting.
    - Used to diagnose diabetes or prediabetes.
    
    **Normal Range:**
    - **≤ 100 mg/dl:** Normal
    - **101-125 mg/dl:** Prediabetes
    - **≥ 126 mg/dl:** Diabetes

    In this model:
    - **FastingBS ≤ 120 mg/dl:** considered normal.
    - **FastingBS > 120 mg/dl:** may indicate elevated sugar levels.
    """)

        fig_fbs = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 100,
            title = {'text': "Fasting Blood Sugar"},
            gauge = {
                'axis': {'range': [50, 200]},
                'steps': [
                    {'range': [50, 100], 'color': "lightgreen"},
                    {'range': [101, 125], 'color': "yellow"},
                    {'range': [126, 200], 'color': "red"}],
                'bar': {'color': "darkgreen"}
            }
        ))
        st.plotly_chart(fig_fbs, use_container_width=True)

    
    with st.expander("❤️ Learn about Max Heart Rate (MaxHR)"):
        st.markdown("""
    **Max Heart Rate (MaxHR):**
    - The maximum number of heart beats per minute during exercise.
    - Used to assess cardiovascular fitness.

    **Formula for estimate:**  
    220 - Age

    - Lower MaxHR during exercise may indicate heart problems.
    """)

        fig_maxhr = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 160,
            title = {'text': "Max Heart Rate"},
            gauge = {
                'axis': {'range': [60, 220]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [60, 100], 'color': "red"},
                    {'range': [100, 140], 'color': "yellow"},
                    {'range': [140, 180], 'color': "lightgreen"},
                    {'range': [180, 220], 'color': "orange"}]
            }
        ))
        st.plotly_chart(fig_maxhr, use_container_width=True)


    with st.expander("🏃‍♂️ Learn about Exercise Induced Angina (ExerciseAngina)"):
        st.markdown("""
    **Exercise Induced Angina (ExerciseAngina):**
    - Chest pain triggered during physical activity.
    - Indicates that the heart isn't receiving enough oxygen.

    In this model:
    - **Yes (1):** Angina present during exercise — higher risk.
    - **No (0):** No angina during exercise — lower risk.
        """)


    with st.expander("📉 Learn about ST Depression (Oldpeak)"):
        st.markdown("""
    **Oldpeak (ST Depression):**
    - Measures ST segment depression induced by exercise.
    - Indicates abnormal heart response during exercise.

    **Values:**
    - **0.0 - 2.0:** Normal or mildly abnormal.
    - **2.0 - 4.0:** Moderately abnormal.
    - **> 4.0:** Severe abnormality.

    Higher Oldpeak values suggest greater ischemia risk.
        """)

        fig_oldpeak = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 1.5,
            title = {'text': "ST Depression (Oldpeak)"},
            gauge = {
                'axis': {'range': [0, 6]},
                'steps': [
                    {'range': [0, 2], 'color': "lightgreen"},
                    {'range': [2, 4], 'color': "orange"},
                    {'range': [4, 6], 'color': "red"}],
                    'bar': {'color': "darkgreen"}
            }
        ))
        st.plotly_chart(fig_oldpeak, use_container_width=True)

    with st.expander("📈 Learn about ST Slope (ST_Slope)"):
        st.markdown("""
    **ST Slope (ST_Slope):**
    - Describes the slope of the peak exercise ST segment.
    - Helps assess severity of ischemia.

    **Categories:**
    - **Upsloping (0):** Usually normal.
    - **Flat (1):** Associated with ischemia.
    - **Downsloping (2):** Strong indicator of heart disease.
        """)


    st.markdown("---")
    st.markdown("""
    ### ℹ️ **Note:**  
    This application is for educational and informational purposes only.  
    Please consult medical professionals for actual diagnosis and treatment.
    """)



