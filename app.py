import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Configure the Streamlit page layout
st.set_page_config(page_title="AI Medical Assistant", layout="centered")

# Cache data loading and training to make the app incredibly fast
@st.cache_resource
def load_and_train_model():
    # Fetch your original training file directly from your GitHub repository path
    data = pd.read_csv("Training.csv")

    
    # Remove unneeded empty tracking columns if present in the data structure
    if "Unnamed: 133" in data.columns:
        data = data.drop(columns=["Unnamed: 133"])
        
    # Isolate symptoms from the target illness label
    X = data.drop(columns=["prognosis"])
    y = data["prognosis"]
    
    # Train model directly on the server container
    model = RandomForestClassifier(random_state=42)
    model.fit(X.values, y)
    
    return model, list(X.columns)

# Safely extract your operational assets
try:
    model, features = load_and_train_model()
    model_ready = True
except Exception as e:
    st.error("Could not find 'Training.csv' on your GitHub repository path. Please verify the file is uploaded to GitHub.")
    model_ready = False

# Mapping database tracking for predictions
DISEASE_INFO = {
    "Fungal infection": {"desc": "A skin condition caused by fungus targeting warm, moist body areas.", "specialist": "Dermatologist"},
    "Allergy": {"desc": "An immune system reaction to a foreign substance or allergen exposure.", "specialist": "Allergist / Immunologist"},
    "GERD": {"desc": "Gastroesophageal reflux disease causes digestive acids to irritate the food pipe.", "specialist": "Gastroenterologist"},
    "Chronic cholestasis": {"desc": "A condition where bile flow from the liver is reduced or stalled.", "specialist": "Hepatologist / Gastroenterologist"},
    "Drug Reaction": {"desc": "An adverse side effect triggered by an ingested or applied medication.", "specialist": "General Physician / Allergist"},
    "Peptic ulcer diseae": {"desc": "Sores that develop on the inner lining of your stomach and upper small intestine.", "specialist": "Gastroenterologist"},
    "AIDS": {"desc": "A chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV).", "specialist": "Infectious Disease Specialist"},
    "Diabetes ": {"desc": "A group of diseases that result in too much sugar in the blood (high blood glucose).", "specialist": "Endocrinologist"},
    "Gastroenteritis": {"desc": "An intestinal infection marked by diarrhea, cramps, nausea, vomiting, and fever.", "specialist": "Gastroenterologist"},
    "Bronchial Asthma": {"desc": "A condition in which a person's airways become inflamed, narrow and swell, and produce extra mucus.", "specialist": "Pulmonologist / Allergist"},
    "Hypertension ": {"desc": "A condition in which the force of the blood against the artery walls is too high.", "specialist": "Cardiologist"},
    "Migraine": {"desc": "A headache of varying intensity, often accompanied by nausea and sensitivity to light and sound.", "specialist": "Neurologist"},
    "Cervical spondylosis": {"desc": "Age-related wear and tear affecting the spinal disks in your neck.", "specialist": "Orthopedic Surgeon / Neurologist"},
    "Paralysis (brain hemorrhage)": {"desc": "Loss of muscle function in part of your body, caused by bleeding inside the brain tissue.", "specialist": "Neurologist / Neurosurgeon"},
    "Jaundice": {"desc": "A yellowing of the skin and eyes caused by high levels of bilirubin.", "specialist": "Hepatologist / Gastroenterologist"},
    "Malaria": {"desc": "A disease caused by a plasmodium parasite, transmitted by the bite of infected mosquitoes.", "specialist": "Infectious Disease Specialist"},
    "Chicken pox": {"desc": "A highly contagious viral infection causing an itchy, blister-like rash on the skin.", "specialist": "General Physician / Pediatrician"},
    "Dengue": {"desc": "A mosquito-borne viral disease occurring in tropical and subtropical areas.", "specialist": "Infectious Disease Specialist"},
    "Typhoid": {"desc": "A bacterial infection spread through contaminated food and water.", "specialist": "Infectious Disease Specialist"},
    "hepatitis A": {"desc": "A highly contagious liver infection caused by the hepatitis A virus.", "specialist": "Hepatologist / Gastroenterologist"},
    "Hepatitis B": {"desc": "A severe liver infection caused by the hepatitis B virus.", "specialist": "Hepatologist / Gastroenterologist"},
    "Hepatitis C": {"desc": "An infection caused by a virus that attacks the liver and leads to inflammation.", "specialist": "Hepatologist / Gastroenterologist"},
    "Hepatitis D": {"desc": "A serious liver disease caused by the hepatitis D virus, requiring concurrent Hepatitis B infection.", "specialist": "Hepatologist / Gastroenterologist"},
    "Hepatitis E": {"desc": "A liver disease caused by the hepatitis E virus, mainly transmitted through contaminated drinking water.", "specialist": "Hepatologist / Gastroenterologist"},
    "Alcoholic hepatitis": {"desc": "Liver inflammation caused by drinking too much alcohol.", "specialist": "Hepatologist / Gastroenterologist"},
    "Tuberculosis": {"desc": "A serious infectious bacterial disease that mainly affects the lungs.", "specialist": "Pulmonologist"},
    "Common Cold": {"desc": "A common viral infection of the nose and throat.", "specialist": "General Physician"},
    "Pneumonia": {"desc": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid.", "specialist": "Pulmonologist / General Physician"},
    "Dimorphic hemmorhoids(piles)": {"desc": "Swollen and inflamed veins in the anus and lower rectum.", "specialist": "General Surgeon / Proctologist"},
    "Heart attack": {"desc": "A medical emergency where blood flow to a part of the heart is blocked.", "specialist": "Cardiologist"},
    "Varicose veins": {"desc": "Gnarled, enlarged veins, most commonly appearing in the legs and feet.", "specialist": "Vascular Surgeon"},
    "Hypothyroidism": {"desc": "A condition in which the thyroid gland doesn't produce enough thyroid hormone.", "specialist": "Endocrinologist"},
    "Hyperthyroidism": {"desc": "The overproduction of a hormone by the butterfly-shaped gland in the neck (thyroid).", "specialist": "Endocrinologist"},
    "Hypoglycemia": {"desc": "An unsafe drop in blood sugar levels, common in diabetes management.", "specialist": "Endocrinologist"},
    "Osteoarthristis": {"desc": "A type of arthritis that occurs when flexible tissue at the ends of bones wears down.", "specialist": "Rheumatologist / Orthopedist"},
    "Arthritis": {"desc": "Inflammation of one or more joints, causing pain and stiffness.", "specialist": "Rheumatologist"},
    "(vertigo) Paroymsal  Positional Vertigo": {"desc": "A sensation of spinning caused by inner ear problems.", "specialist": "ENT Specialist / Neurologist"},
    "Acne": {"desc": "A skin condition that occurs when hair follicles become plugged with oil and dead skin cells.", "specialist": "Dermatologist"},
    "Urinary tract infection": {"desc": "An infection in any part of the urinary system, including kidneys, ureters, bladder, and urethra.", "specialist": "Urologist / General Physician"},
    "Psoriasis": {"desc": "A condition in which skin cells build up and form scales and itchy, dry patches.", "specialist": "Dermatologist"},
    "Impetigo": {"desc": "A highly contagious skin infection that causes sores, mainly around the nose and mouth.", "specialist": "Dermatologist / General Physician"}
}

st.title("🩺 AI Health Diagnosis Assistant")
st.write("Select your symptoms below to get an algorithmic prediction model evaluation.")

if model_ready:
    # Process features into clean drop-down titles
    clean_features = [f.replace("_", " ").title() for f in features]
    selected_clean = st.multiselect("Choose Symptoms:", clean_features)

    if st.button("Predict Diagnosis", type="primary"):
        if not selected_clean:
            st.warning("Please select at least one symptom.")
        else:
            # Build your binary evaluation vector input
            input_data = np.zeros(len(features))
            for clean_sym in selected_clean:
                raw_name = clean_sym.lower().replace(" ", "_")
                if raw_name in features:
                    idx = features.index(raw_name)
                    input_data[idx] = 1
            
            # Predict the structural target string
            prediction = model.predict([input_data])[0]
            
            st.success(f"### Predicted Condition: **{prediction}**")
            
            # Surface mapped context variables from knowledge engine dictionary
            if prediction in DISEASE_INFO:
                info = DISEASE_INFO[prediction]
                st.info(f"**Description:** {info['desc']}")
                st.warning(f"🎯 **Recommended Department:** You should consult a **{info['specialist']}**.")
            else:
                st.info("**Description:** Information profile pending expansion.")
                st.warning("🎯 **Recommended Department:** Please consult a General Physician for primary routing.")

st.markdown("---")
st.caption("⚠️ **Educational Project Disclaimer:** This software functions strictly as a data-science exercise using training datasets. It does not replace professional medical evaluations, diagnoses, or clinical treatment plans.")
