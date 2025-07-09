import streamlit as st
import pickle
import numpy as np

# Load models and scalers
diabetes_model = pickle.load(open('D:\\disease prediction\\new one\\diabetes_model.pk1', 'rb'))
diabetes_scaler = pickle.load(open("D:\disease prediction\\new one\scaler_of_diabetes.pk1", 'rb'))

heart_model = pickle.load(open("D:\disease prediction\\new one\heart_model.pk1", 'rb'))
heart_scaler = pickle.load(open('D:\\disease prediction\\new one\\scaler_of_heart_model.sav', 'rb'))

parkinsons_model = pickle.load(open("D:\\disease prediction\\new one\\parkinson_model.pk1", 'rb'))
parkinsons_scaler = pickle.load(open('D:\\disease prediction\\new one\\scaler_of_parkinsons.pkl', 'rb'))

# App UI
st.set_page_config(page_title="Disease Prediction App", layout="centered")
st.title("ML-Based Disease Prediction System")
st.write("Choose a disease from the sidebar and enter the required health data to get a prediction.")

# Sidebar Navigation
selected_disease = st.sidebar.selectbox("Choose Disease", ["Diabetes", "Heart Disease", "Parkinson's"])

# Prediction Functions
def predict_diabetes(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    scaled = diabetes_scaler.transform(input_array)
    prediction = diabetes_model.predict(scaled)
    return 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

def predict_heart(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    scaled = heart_scaler.transform(input_array)
    prediction = heart_model.predict(scaled)
    return 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'

def predict_parkinsons(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    scaled = parkinsons_scaler.transform(input_array)
    prediction = parkinsons_model.predict(scaled)
    return "Parkinson's Detected" if prediction[0] == 1 else "No Parkinson's"

# Diabetes Form (BRFSS)
if selected_disease == "Diabetes":
    st.subheader("Diabetes Prediction")
    HighBP = st.selectbox('High Blood Pressure (1=Yes, 0=No)', [1, 0])
    HighChol = st.selectbox('High Cholesterol (1=Yes, 0=No)', [1, 0])
    CholCheck = st.selectbox('Cholesterol Check in last 5 years (1=Yes, 0=No)', [1, 0])
    BMI = st.number_input('BMI', min_value=0.0)
    Smoker = st.selectbox('Smoked 100+ cigarettes (1=Yes, 0=No)', [1, 0])
    Stroke = st.selectbox('Ever had a stroke (1=Yes, 0=No)', [1, 0])
    HeartDiseaseorAttack = st.selectbox('Heart Disease or Heart Attack (1=Yes, 0=No)', [1, 0])
    PhysActivity = st.selectbox('Physical Activity in past 30 days (1=Yes, 0=No)', [1, 0])
    Fruits = st.selectbox('Consumes Fruit 1+ times/day (1=Yes, 0=No)', [1, 0])
    AnyHealthcare = st.selectbox('Any healthcare coverage (1=Yes, 0=No)', [1, 0])
    NoDocbcCost = st.selectbox('Could not see doctor due to cost (1=Yes, 0=No)', [1, 0])
    GenHlth = st.selectbox('General Health (1=Excellent to 5=Poor)', [1, 2, 3, 4, 5])
    MentHlth = st.number_input('Mental Health - bad days (last 30 days)', min_value=0)
    PhysHlth = st.number_input('Physical Health - bad days (last 30 days)', min_value=0)
    DiffWalk = st.selectbox('Difficulty Walking (1=Yes, 0=No)', [1, 0])
    Sex = st.selectbox('Sex (1=Male, 0=Female)', [1, 0])
    Age = st.selectbox('Age Category (1 to 13)', list(range(1, 14)))
    Education = st.selectbox('Education Level (1 to 6)', list(range(1, 7)))
    Income = st.selectbox('Income Category (1 to 8)', list(range(1, 9)))

    if st.button('Predict'):
        input_data = [HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                      PhysActivity, Fruits, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth,
                      PhysHlth, DiffWalk, Sex, Age, Education, Income]
        result = predict_diabetes(input_data)
        st.success(result)

# Heart Disease Form
elif selected_disease == "Heart Disease":
    st.subheader("Heart Disease Prediction")
    male = st.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
    age = st.number_input('Age', min_value=0)
    education = st.selectbox('Education Level (1 to 4)', [1, 2, 3, 4])
    currentSmoker = st.selectbox('Current Smoker (1 = Yes, 0 = No)', [1, 0])
    cigsPerDay = st.number_input('Cigarettes per Day', min_value=0)
    BPMeds = st.selectbox('On BP Medication (1 = Yes, 0 = No)', [1, 0])
    prevalentStroke = st.selectbox('Had a Stroke Before (1 = Yes, 0 = No)', [1, 0])
    prevalentHyp = st.selectbox('History of Hypertension (1 = Yes, 0 = No)', [1, 0])
    diabetes = st.selectbox('Diabetic (1 = Yes, 0 = No)', [1, 0])
    totChol = st.number_input('Total Cholesterol', min_value=0.0)
    sysBP = st.number_input('Systolic BP', min_value=0.0)
    diaBP = st.number_input('Diastolic BP', min_value=0.0)
    BMI = st.number_input('BMI', min_value=0.0)
    heartRate = st.number_input('Heart Rate', min_value=0.0)
    glucose = st.number_input('Glucose', min_value=0.0)

    if st.button('Predict'):
        input_data = [male, age, education, currentSmoker, cigsPerDay, BPMeds,
                      prevalentStroke, prevalentHyp, diabetes, totChol,
                      sysBP, diaBP, BMI, heartRate, glucose]
        result = predict_heart(input_data)
        st.success(result)

# Parkinson's Form
elif selected_disease == "Parkinson's":
    st.subheader("Parkinson's Disease Prediction")
    fo = st.number_input('MDVP:Fo(Hz)')
    fhi = st.number_input('MDVP:Fhi(Hz)')
    flo = st.number_input('MDVP:Flo(Hz)')
    jitter_percent = st.number_input('MDVP:Jitter(%)')
    jitter_abs = st.number_input('MDVP:Jitter(Abs)')
    rap = st.number_input('MDVP:RAP')
    ppq = st.number_input('MDVP:PPQ')
    ddp = st.number_input('Jitter:DDP')
    shimmer = st.number_input('MDVP:Shimmer')
    shimmer_db = st.number_input('MDVP:Shimmer(dB)')
    apq3 = st.number_input('Shimmer:APQ3')
    apq5 = st.number_input('Shimmer:APQ5')
    apq = st.number_input('MDVP:APQ')
    dda = st.number_input('Shimmer:DDA')
    nhr = st.number_input('NHR')
    hnr = st.number_input('HNR')
    rpde = st.number_input('RPDE')
    dfa = st.number_input('DFA')
    spread1 = st.number_input('Spread1')
    spread2 = st.number_input('Spread2')
    d2 = st.number_input('D2')
    ppe = st.number_input('PPE')

    if st.button('Predict'):
        input_data = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                      shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                      rpde, dfa, spread1, spread2, d2, ppe]
        result = predict_parkinsons(input_data)
        st.success(result)

