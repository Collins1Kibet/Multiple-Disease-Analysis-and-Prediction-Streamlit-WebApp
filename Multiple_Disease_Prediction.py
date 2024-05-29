import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu



working_directory = os.path.dirname(os.path.abspath(__file__))


allergy_cold_covid_or_flu_model_path = os.path.join(working_directory, "Trained_Models", "Trained_Model_Allergy_Cold_Covid_or_Flu.sav")
diabetes_model_path = os.path.join(working_directory, "Trained_Models", "Trained_Model_Diabetes.sav")
heart_disease_model_path = os.path.join(working_directory, "Trained_Models", "Trained_Model_heart_disease.sav")
parkisons_disease_model_path = os.path.join(working_directory, "Trained_Models", "Trained_Model_Parkinsons_Disease.sav")
breast_cancer_model_path = os.path.join(working_directory, "Trained_Models", "Trained_Model_Breast_Cancer.sav")
feature_names_file_path = os.path.join(working_directory, "Trained_Models", "feature_names.pkl")

allergy_cold_covid_or_flu_model = pickle.load(open(allergy_cold_covid_or_flu_model_path, 'rb'))
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
heart_disease_model = pickle.load(open(heart_disease_model_path, 'rb'))
parkisons_disease_model = pickle.load(open(parkisons_disease_model_path, 'rb'))
breast_cancer_model = pickle.load(open(breast_cancer_model_path, 'rb'))
feature_names = pickle.load(open(feature_names_file_path, 'rb'))


st.set_page_config(
    page_title="Cobest Kenya Multiple Disease Prediction System",
    page_icon='🩺',
    layout= 'wide'
)

with st.sidebar:
    selected = option_menu(menu_title= "Cobest Kenya Multiple Disease Prediction System",
                           options= ['Allergy, Cold, Covid or Flu Detection',
                                     'Diabetes Detection',
                                     'Heart Disease Detection',
                                     "Parkinson's Disease Detection",
                                     "Breast Cancer Tumor Classification"],
                                     menu_icon= '👨🏾‍⚕️', icons= ['emoji-grimace', 'activity', 'heart', 'person-dash', 'x-circle'],
                                     default_index=0)
    

if (selected == "Allergy, Cold, Covid or Flu Detection"):
    st.title("Allergy, Cold, Covid or Flu Analysis and Detection")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        COUGH = st.text_input("Are you Coughing? 1 = Yes and 0 = No")
    with col2:
        MUSCLE_ACHES = st.text_input("Experiencing Muscle-Aches? (1 = 'Yes' and 0 = 'No')")
    with col3:
        TIREDNESS = st.text_input("Having a feeling of Tiredness? (1 = 'Yes' and 0 = 'No')")  
    with col4:
        SORE_THROAT = st.text_input("Having a Sore Throat? (1 = 'Yes' and 0 = 'No')")
    with col1:
        RUNNY_NOSE = st.text_input("Having a Runny Nose (1 = 'Yes' and 0 = 'No')")
    with col2:
        STUFFY_NOSE = st.text_input("Having a Stuffy Nose? (1 = 'Yes' and 0 = 'No')")
    with col3:
        FEVER = st.text_input("Experiencing Fever? (1 = 'Yes' and 0 = 'No')")
    with col4:
        NAUSEA = st.text_input("Feeling Nausea? (1 = 'Yes' and 0 = 'No')")
    with col1:
        VOMITING = st.text_input("Vommiting? (1 = 'Yes' and 0 = 'No')")
    with col2:
        DIARRHEA = st.text_input("Experiencing Diarrhea? (1 = 'Yes and 0 = 'No')")
    with col3:
        SHORTNESS_OF_BREATH = st.text_input("Experiencing Shortness of Breath? (1 = 'Yes and 0 = 'No')")  
    with col4:
        DIFFICULTY_BREATHING = st.text_input("Having Really Difficulty Breathing? (1 = 'Yes' and 0 = 'No')")
    with col1:
        LOSS_OF_TASTE = st.text_input("Lost a Sense of Taste? (1 = 'Yes' and 0 = 'No')")
    with col2:
        LOSS_OF_SMELL = st.text_input("lost a Sense of Smell? (1 = 'Yes' and 0 = 'No')")
    with col3:
        ITCHY_NOSE = st.text_input("Having a Itchy Nose? (1 = 'Yes' and 0 = 'No')")
    with col4:
        ITCHY_EYES = st.text_input("Having Itchy Eyes? (1 = 'Yes' and 0 = 'No')")
    with col1:
        ITCHY_MOUTH = st.text_input("Experiencing Itchiness in and around the Mouth? (1 = 'Yes' and 0 = 'No')")
    with col2:
        ITCHY_INNER_EAR = st.text_input("Experiencing Itchiness in the Inner Ear? (1 = 'Yes' and 0 = 'No')")
    with col3:
        SNEEZING = st.text_input("Are you Sneezing? (1 = 'Yes' and 0 = 'No')")
    with col4:
        PINK_EYE = st.text_input("Having a Pink Eye? (1 = 'Yes' and 0 = 'No')")

    ACCF_diagnosis = ''

    if st.button('Your Test Results'):
        input_data = {
            'COUGH': COUGH, 'MUSCLE_ACHES': MUSCLE_ACHES, 'TIREDNESS': TIREDNESS, 'SORE_THROAT': SORE_THROAT, 
            'RUNNY_NOSE': RUNNY_NOSE, 'STUFFY_NOSE': STUFFY_NOSE, 'FEVER': FEVER, 'NAUSEA': NAUSEA, 
            'VOMITING': VOMITING, 'DIARRHEA': DIARRHEA, 'SHORTNESS_OF_BREATH': SHORTNESS_OF_BREATH, 
            'DIFFICULTY_BREATHING': DIFFICULTY_BREATHING, 'LOSS_OF_TASTE': LOSS_OF_TASTE, 
            'LOSS_OF_SMELL': LOSS_OF_SMELL, 'ITCHY_NOSE': ITCHY_NOSE, 'ITCHY_EYES': ITCHY_EYES, 
            'ITCHY_MOUTH': ITCHY_MOUTH, 'ITCHY_INNER_EAR': ITCHY_INNER_EAR, 'SNEEZING': SNEEZING, 'PINK_EYE': PINK_EYE
        }

        input_df = pd.DataFrame([input_data], columns=feature_names)

        allergy_cold_covid_or_Flu_prediction = allergy_cold_covid_or_flu_model.predict(input_df)

        if allergy_cold_covid_or_Flu_prediction[0] == 0:
            ACCF_diagnosis = 'The Person has Allergy'
        elif allergy_cold_covid_or_Flu_prediction[0] == 1:
            ACCF_diagnosis = 'The Person has Cold'
        elif allergy_cold_covid_or_Flu_prediction[0] == 2:
            ACCF_diagnosis = 'The Person has Covid-19'
        else:
            ACCF_diagnosis = 'The Person has Flu'

    st.success(ACCF_diagnosis)


    
if (selected == "Diabetes Detection"):
    st.title("Diabetes Detection")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Prenancies")
    with col2:
        Glucose = st.text_input("Glucose Level (mg/dL)")
    with col3:
        BloodPressure = st.text_input("Blood Pressure Rate (mmHg)")
    with col1:
        SkinThickness = st.text_input("Skin Thickness (mm)")
    with col2:
        Insulin = st.text_input("Insulin Level (µU/mL)")
    with col3:
        BMI = st.text_input("Measure of BMI (kg/m²)")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pediree Function Percentage")
    with col2:
        Age = st.text_input("How Old is the Person")

    diabetes_diagnosis = ''

    if st.button("Diabetes Test Results"):
        diabetes_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, 
                                                       DiabetesPedigreeFunction, Age]])

        if (diabetes_prediction[0] == 1):
            diabetes_diagnosis = 'The Person is Diabetic'
        else:
            diabetes_diagnosis = 'The Person is not Diabetic'

    st.success(diabetes_diagnosis)



if (selected == "Heart Disease Detection"):
    st.title("Heart Disease Detection")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age  = st.number_input("Age of the Person")
    with col2:
        sex  = st.number_input("Sex of the Person")
    with col3:
        cp  = st.number_input("Chest Pain measured")
    with col4:
        trestbps = st.number_input("Resting Blood Pressure Measured")
    with col1:
        chol = st.number_input("Serum Cholestoral mg/dl")
    with col2:
        fbs = st.number_input("Fasting Blood Sugar 120mg/dl")
    with col3:
        restecg = st.number_input("Resting Electrocardiograpic Results (Values 0,1,2)")
    with col4:
        thalach = st.number_input("Maximum Heart Rate Achieved")
    with col1:
        exang = st.number_input("Exercise Induced Angina")
    with col2:
        oldpeak = st.number_input("oldpeak = ST Depression Induced by exercise relative to rest")
    with col3:
        slope = st.number_input("The slope of the Peak Exercise ST Segment")
    with col4:
        ca = st.number_input("The number of Major Vessels(0-3) covered by Flourosopy")
    with col1:
        thal = st.number_input("thal: 3 = Normal; 6 = Fixed defect; 7 = Reversable defect")

    heart_disease_diagnosis = ''

    if st.button("Heart Disease Test Result"):
        heart_disease_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, 
                                                                 slope, ca, thal]])

        if (heart_disease_prediction[0]==1):
            heart_disease_diagnosis = "The Person has Heart Disease"
        else:
            heart_disease_diagnosis = "The Person does not have Heart Disease"

    st.success(heart_disease_diagnosis)


    
if (selected == "Parkinson's Disease Detection"):
    st.title("Parkinson's Disease Detection")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        MDVP_Fo_Hz = st.text_input('Average Vocal Fundamental Frequency')
    with col2:
        MDVP_Fhi_Hz = st.text_input('Maximum Vocal Fundamental frequency')
    with col3:
        MDVP_Flo_Hz = st.text_input('Minimum Vocal Fundamental Frequency')
    with col4:
        MDVP_Jitter = st.text_input('Measure of Variation in Fundamental Frequency')
    with col1:
        MDVP_Jitter_Abs = st.text_input('Measure of Jitter Variation in Abs Frequency')
    with col2:
        MDVP_RAP = st.text_input('Measure of MDVP Variation in RAP Frequency')
    with col3:
        MDVP_PPQ = st.text_input('Measure of MDVP Variation in PPQ Frequency')
    with col4:
        Jitter_DDP = st.text_input('Measure of Jitter Variation in DDP Frequency')
    with col1:
        MDVP_Shimmer = st.text_input('Measure of Shimmer Variation in Amplitude')
    with col2:
        MDVP_Shimmer_dB = st.text_input('Measure of Shimmer Variation in dB Amplitude')
    with col3:
        Shimmer_APQ3 = st.text_input('Measure of Shimmer Variation in APQ3 Amplitude')
    with col4:
        Shimmer_APQ5 = st.text_input('Measure of Shimmer Variation in APQ5 Amplitude')
    with col1:
        MDVP_APQ = st.text_input('Measure of MDVP Variation in APQ Amplitude')
    with col2:
        Shimmer_DDA = st.text_input('Measure of Shimmer Variation in DDA Amplitude')
    with col3:
        NHR = st.text_input('NHR measure of ratio of noise to tonal components in the voice')
    with col4:
        HNR = st.text_input('HNR easure of ratio of noise to tonal components in the voice')
    with col1:
        RPDE = st.text_input('Two nonlinear dynamical complexity measures')
    with col2:
        DFA = st.text_input('Signal fractal scaling exponent')
    with col3:
        spread1 = st.text_input('First nonlinear measures of fundamental frequency variation')
    with col4:
        spread2 = st.text_input('Second nonlinear measures of fundamental frequency variation')
    with col1:
        D2 = st.text_input('D2 nonlinear measures of fundamental frequency variation')
    with col2:
        PPE = st.text_input('PPE nonlinear measures of fundamental frequency variation')
        
    parkinsons_disease_diagnosis = ''

    if st.button("Parkinson's Disease Test Result"):
        parkinsons_disease_prediction = parkisons_disease_model.predict([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
        
        if (parkinsons_disease_prediction[0] == 0):
            parkinsons_disease_diagnosis = "The Person does not have Parkinson's Disease"
        else:
            parkinsons_disease_diagnosis = "The Person has Parkinson's Disease"

    st.success(parkinsons_disease_diagnosis)



if (selected == "Breast Cancer Tumor Classification"):
    st.title("Breast Cancer Tumor Classification (units: mm, g)")

    col1, col2, col3 = st.columns(3)

    with col1:
        mean_radius = st.text_input("Mean Radius of the Tumor")
    with col2:
        radius = st.text_input("Radius Error of the Tumor")
    with col3:
        worst_radius = st.text_input("The Worst Radius of the Tumor")
    with col1:
        mean_texture = st.text_input("Measured Mean Texture of the Tumor")
    with col2:
        perimeter = st.text_input("The Perimeter Error of the Tumor")
    with col3:
        worst_texture = st.text_input("The Worst Texture of the Tumor")
    with col1:
        mean_perimeter = st.text_input("Mean Perimeter of the Tumor")
    with col2:
        area = st.text_input("The Area Error of the Tumor")
    with col3:
        worst_perimeter = st.text_input("The Worst Perimeter of the Tumor")
    with col1:
        mean_area = st.text_input("Mean Area of the Tumor")
    with col2:
        smoothness = st.text_input("The Smoothness Error of the Tumor")
    with col3:
        worst_area = st.text_input("The Worst Area of the Tumor")
    with col1:
        mean_smoothness = st.text_input("The Smoothness of the Tumor")
    with col2:
        compactness = st.text_input("The Compactness Error of the Tumor")
    with col3:
        worst_smoothness = st.text_input("The Worst Smoothness of the Tumor")
    with col1:
        mean_compactness = st.text_input("Mean Compactness of the Tumor")
    with col2:
        concavity = st.text_input("The Concavity Error of the Tumor")
    with col3:
        worst_compactness = st.text_input("The Worst Compactness of the Tumor")
    with col1:
        mean_concavity = st.text_input("Mean Concavity of the Tumor")
    with col2:
        concave_points = st.text_input("The Concave Points Error of the Tumor")
    with col3:
        worst_concavity = st.text_input("The Worst Concavity of the Tumor") 
    with col1:
        mean_concave_points = st.text_input("Mean Concave Points of the Tumor")
    with col2:
        symmetry = st.text_input("The Symmetry Error of the Tumor")
    with col3:
        worst_concave_points = st.text_input("The Worst Concave Points of the Tumor")
    with col1:
        mean_symmetry = st.text_input("The Mean Symmetry of the Tumor")
    with col2:
        fractal_dimension = st.text_input("The Fractal Dimension Error of the Tumor")
    with col3:
        worst_symmetry = st.text_input("The Worst Symmetry of the Tumor")
    with col1:
        mean_fractal_dimension = st.text_input("The Mean Fractal Dimension of the Tumor")
    with col2:
        texture = st.text_input("Texture Error of the Tumor")
    with col3:
        worst_fractal_dimension = st.text_input("The Worst Fractal Dimension of the Tumor")


    breast_cancer_diagnosis = ''

    if st.button("Test Result"):
        breast_cancer_tumor_classification = breast_cancer_model.predict([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, 
                                          mean_concave_points, mean_symmetry, mean_fractal_dimension,radius, texture, perimeter, area, smoothness, 
                                          compactness, concavity, concave_points, symmetry, fractal_dimension, worst_radius, worst_texture,
                                          worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points,
                                          worst_symmetry, worst_fractal_dimension]])
            
        if (breast_cancer_tumor_classification[0] == 0):
            breast_cancer_diagnosis = "The Tumor is Malignant"
        else:
            breast_cancer_diagnosis = "The Tumor is Benign"

    st.success(breast_cancer_diagnosis)
