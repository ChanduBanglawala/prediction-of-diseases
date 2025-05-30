
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np

# Load pre-trained models
diabetes_model = pickle.load(open("D:\disease prediction\diabetes.sav", "rb"))
heart_disease_model = pickle.load(open("D:\disease prediction\heart.sav","rb"))
parkinsons_model = pickle.load(open("D:\disease prediction\parkinson.sav", "rb"))

with st.sidebar:
  
    selected = option_menu(
    'PREDICTION OF DISEASE OUTBREAK SYSTEM',
    ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
    icons=['droplet', 'heart', 'person'],
    default_index=0)

if selected=='Diabetes Prediction':
    st.title('Diabetes Prediction using Machine Learning')
    col1,col2,col3,col4=st.columns(4)
    with col1:
        pregnancies=st.text_input('Number of pregnancies')
    with col2:
        Glucose=st.text_input('Glucose Level')
    with col3:
        Bloodpressure=st.text_input('Blood pressure value')
    with col4:
        SkinThickness=st.text_input('Skin Thickness value')
    with col1:
        Insulin=st.text_input('Insulin Level')
    with col2:
        BMI=st.text_input('BMI value')
    with col3:
        DiabetesPedigreeFunction=st.text_input('Diabetes pedigree Function value')
    with col4:
        Age=st.text_input('Age of the person')

    diab_diagnosis=''
    if st.button('Diabetes Test Result'):
         user_input=[pregnancies,Glucose,Bloodpressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
         user_input=[float(i) for i in user_input]
         diab_prediction=diabetes_model.predict([user_input])
         if diab_prediction[0]==1:
            diab_diagnosis='The person is diabetic'
         else:
            diab_diagnosis='The person is not diabetic'
    st.success(diab_diagnosis)        


if selected=='Heart Disease Prediction':
    st.title('Heart Disease Prediction using Machine Learninng')
    col1,col2,col3,col4=st.columns(4)
    with col1:
        age=st.text_input('Enter the patient age')
    with col2:
        sex=st.text_input('Gender for M-1 & F-0') 
    with col3:
        cp=st.text_input('Constrictive Pericarditis(Cp)')
    with col4:
        trestbps=st.text_input('Trestbps value')
    with col1:
        chol=st.text_input('Cholesterol Value')
    with col2:
        fbs=st.text_input('fbs value')
    with col3:
        restecg=st.text_input('restecg value')
    with col4:
        thalach=st.text_input('thalach Value')
    with col1:
        exang=st.text_input('exang Value')
    with col2:
        oldpeak=st.text_input('oldpeak value')
    with col2:
        slope=st.text_input('slope Value')
    with col3:
        ca=st.text_input('ca value')
    with col4:
        thal=st.text_input('thal value')
        

    heaart_diagnosis=''
    if st.button('Heart Test Result'):
         user_input=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
         user_input=[float(i) for i in user_input]
         diab_prediction=diabetes_model.predict([user_input])
         if diab_prediction[0]==1:
            diab_diagnosis='The person is diabetic'
         else:
            diab_diagnosis='The person is not diabetic'
    st.success(heaart_diagnosis)        
    
# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

       

