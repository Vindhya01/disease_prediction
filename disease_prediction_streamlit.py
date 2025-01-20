import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from joblib import load

'''# Function to load model from raw URL
def load_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        st.error(f"Failed to fetch the model. HTTP Status code: {response.status_code}")
        return None

# Function to load CSV files from raw URLs
def load_csv(url):
    return pd.read_csv(url)
'''

# Load the model
model = load('./decision_tree_model.joblib')


# Load the disease description and precautions
desc = load_csv("./disease_symptom_dataset//symptom_description.csv")
prec = load_csv("./disease_symptom_dataset//symptom_precaution.csv")

# List of diseases and symptoms (same as your existing lists)
diseases = [
    '(vertigo) Paroymsal Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
    'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis', 'Common Cold', 'Dengue', 'Diabetes', 'Dimorphic hemorrhoids(piles)',
    'Drug Reaction', 'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
    'Hepatitis E', 'Hypertension', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
    'Osteoarthritis', 'Paralysis (brain hemorrhage)', 'Peptic ulcer disease', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid',
    'Urinary tract infection', 'Varicose veins', 'hepatitis A'
]

symptoms = [
    'Itching', 'Skin Rash', 'Nodal Skin Eruptions', 'Continuous Sneezing', 'Shivering', 'Chills', 'Joint Pain', 'Stomach Pain', 'Acidity',
    'Ulcers on Tongue', 'Muscle Wasting', 'Vomiting', 'Burning Micturition', 'Fatigue', 'Weight Gain', 'Anxiety', 'Cold Hands and Feets',
    'Mood Swings', 'Weight Loss', 'Restlessness', 'Lethargy', 'Patches in Throat', 'Irregular Sugar Level', 'Cough', 'High Fever',
    'Sunken Eyes', 'Breathlessness', 'Sweating', 'Dehydration', 'Indigestion', 'Headache', 'Yellowish Skin', 'Dark Urine', 'Nausea',
    'Loss of Appetite', 'Pain Behind the Eyes', 'Back Pain', 'Constipation', 'Abdominal Pain', 'Diarrhoea', 'Mild Fever', 'Yellow Urine',
    'Yellowing of Eyes', 'Acute Liver Failure', 'Fluid Overload', 'Swelling of Stomach', 'Swelled Lymph Nodes', 'Malaise',
    'Blurred and Distorted Vision', 'Phlegm', 'Throat Irritation', 'Redness of Eyes', 'Sinus Pressure', 'Runny Nose', 'Congestion',
    'Chest Pain', 'Weakness in Limbs', 'Fast Heart Rate', 'Pain During Bowel Movements', 'Pain in Anal Region', 'Bloody Stool',
    'Irritation in Anus', 'Neck Pain', 'Dizziness', 'Cramps', 'Bruising', 'Obesity', 'Swollen Legs', 'Swollen Blood Vessels', 'Puffy Face and Eyes',
    'Enlarged Thyroid', 'Brittle Nails', 'Swollen Extremities', 'Excessive Hunger', 'Extra Marital Contacts', 'Drying and Tingling Lips',
    'Slurred Speech', 'Knee Pain', 'Hip Joint Pain', 'Muscle Weakness', 'Stiff Neck', 'Swelling Joints', 'Movement Stiffness', 'Spinning Movements',
    'Loss of Balance', 'Unsteadiness', 'Weakness of One Body Side', 'Loss of Smell', 'Bladder Discomfort', 'Continuous Feel of Urine',
    'Passage of Gases', 'Internal Itching', 'Toxic Look (Typhos)', 'Depression', 'Irritability', 'Muscle Pain', 'Altered Sensorium',
    'Red Spots Over Body', 'Belly Pain', 'Abnormal Menstruation', 'Watering from Eyes', 'Increased Appetite', 'Polyuria', 'Family History',
    'Mucoid Sputum', 'Rusty Sputum', 'Lack of Concentration', 'Visual Disturbances', 'Receiving Blood Transfusion', 'Receiving Unsterile Injections',
    'Coma', 'Stomach Bleeding', 'Distention of Abdomen', 'History of Alcohol Consumption', 'Blood in Sputum', 'Prominent Veins on Calf',
    'Palpitations', 'Painful Walking', 'Pus Filled Pimples', 'Blackheads', 'Scurring', 'Skin Peeling', 'Silver-Like Dusting', 'Small Dents in Nails',
    'Inflammatory Nails', 'Blister', 'Red Sore Around Nose', 'Yellow Crust Ooze'
]

# Streamlit UI components
st.title("Disease Prediction Based on Symptoms")

# Get user input for symptoms (dropdown menu for each symptom)
user_symptoms = st.multiselect("Select your symptoms:", options=symptoms)

# Create feature vector
features = [0] * 222
for symptom in user_symptoms:
    index = symptoms.index(symptom)
    features[index] = 1

# Model prediction and output
if st.button('Predict'):
    # Get prediction probabilities
    proba = model.predict_proba([features])

    # Get the top disease prediction
    top_idx = np.argmax(proba[0])
    top_proba = proba[0][top_idx]
    top_disease = diseases[top_idx]

    # Get disease description
    disp = desc[desc['Disease'] == top_disease].values[0][1] if top_disease in desc["Disease"].unique() else "No description available"

    # Get disease precautions
    precautions = []
    if top_disease in prec["Disease"].unique():
        c = np.where(prec['Disease'] == top_disease)[0][0]
        for j in range(1, len(prec.iloc[c])):
            precautions.append(prec.iloc[c, j])

    # Display the result
    st.subheader(f"Disease Name: {top_disease}")
    st.write(f"Probability: {top_proba:.2f}")
    st.write(f"Disease Description: {disp}")
    st.write("Recommended Things to do at home:")
    for precaution in precautions:
        st.write(f"- {precaution}")
