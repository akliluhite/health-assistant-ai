import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set layout configurations with a dedicated sidebar collapse feature
st.set_page_config(
    page_title="AI Healthcare System Portal",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom High-Quality CSS Styling
st.markdown("""
    <style>
    /* Premium Clinical Soft Light Grey Background tint */
    .stApp {
        background-color: #f1f5f9 !important;
    }
    
    /* Clean Dark Slate Minimalist Headers */
    h1 {
        color: #0f172a !important;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        margin-bottom: 5px !important;
    }
    
    /* Subtitle text formatting styling */
    .sub-heading {
        color: #475569 !important;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Modern minimalist dashboard container card styling overrides */
    div[data-testid="stVerticalBlock"] > div:has(div.stMultiSelect) {
        background: #ffffff !important;
        padding: 30px !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05) !important;
    }

    /* Sleek Clean Primary Button Override */
    .stButton>button[data-testid="baseButton-primary"] {
        background-color: #0284c7 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: background 0.2s ease;
    }
    .stButton>button[data-testid="baseButton-primary"]:hover {
        background-color: #0369a1 !important;
    }

    /* Custom Modern Clean Metrics Components */
    .metric-box {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        border-left: 5px solid #0284c7;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.4rem;
        color: #0f172a;
        font-weight: 700;
        margin-top: 4px;
    }
    .info-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        color: #334155;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading and training structure
@st.cache_resource
def load_and_train_model():
    data = pd.read_csv("Training.csv")
    if "Unnamed: 133" in data.columns:
        data = data.drop(columns=["Unnamed: 133"])
    X = data.drop(columns=["prognosis"])
    y = data["prognosis"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X.values, y)
    return model, list(X.columns)

try:
    model, features = load_and_train_model()
    model_ready = True
except Exception as e:
    st.error("Could not load 'Training.csv' locally. Please verify your file setup.")
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
    "Hyperthyroidism": {"desc": "The overproduction of a hormone by the butterfly-shaped gland in the neck (the thyroid).", "specialist": "Endocrinologist"},
    "Hypoglycemia": {"desc": "An unsafe drop in blood sugar levels, common in diabetes management.", "specialist": "Endocrinologist"},
    "Osteoarthristis": {"desc": "A type of arthritis that occurs when flexible tissue at the ends of bones wears down.", "specialist": "Rheumatologist / Orthopedist"},
    "Arthritis": {"desc": "Inflammation of one or more joints, causing pain and stiffness.", "specialist": "Rheumatologist"},
    "(vertigo) Paroymsal  Positional Vertigo": {"desc": "A sensation of spinning caused by inner ear problems.", "specialist": "ENT Specialist / Neurologist"},
    "Acne": {"desc": "A skin condition that occurs when hair follicles become plugged with oil and dead skin cells.", "specialist": "Dermatologist"},
    "Urinary tract infection": {"desc": "An infection in any part of the urinary system, including kidneys, ureters, bladder, and urethra.", "specialist": "Urologist / General Physician"},
    "Psoriasis": {"desc": "A condition in which skin cells build up and form scales and itchy, dry patches.", "specialist": "Dermatologist"},
    "Impetigo": {"desc": "A highly contagious skin infection that causes sores, mainly around the nose and mouth.", "specialist": "Dermatologist / General Physician"}
}

# Clean layout application presentation
st.title("Clinical Diagnostic Interface")
st.markdown("<div class='sub-heading'>Enter clinical symptom indicators below to query the prediction evaluation workspace.</div>", unsafe_allow_html=True)

if model_ready:
    clean_features = [f.replace("_", " ").title() for f in features]
    
    # Session state trick to manage clear layout button actions
    if "symptom_key" not in st.session_state:
        st.session_state.symptom_key = 0

    selected_clean = st.multiselect(
        "Identify Observed Symptoms:", 
        clean_features, 
        placeholder="Type to filter symptoms...",
        key=f"symptoms_{st.session_state.symptom_key}"
    )
    
    # Layout configuration grid columns
    btn_col1, btn_col2 = st.columns([4, 1])
    
    with btn_col1:
        submit_btn = st.button("Run Diagnostic Analysis", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Reset", use_container_width=True):
            st.session_state.symptom_key += 1
            st.rerun()

    if submit_btn:
        if not selected_clean:
            st.warning("Please identify at least one symptom indicator before running analysis.")
        else:
            input_data = np.zeros(len(features))
            for clean_sym in selected_clean:
                raw_name = clean_sym.lower().replace(" ", "_")
                if raw_name in features:
                    idx = features.index(raw_name)
                    input_data[idx] = 1
            
            # Predict core condition text
            prediction_array = model.predict([input_data])
            prediction = str(prediction_array[0]).strip()
            
            # Feature 1: Model Certainty Probability Array Scoring
            probabilities = model.predict_proba([input_data])[0]
            class_idx = np.where(model.classes_ == prediction)[0][0]
            confidence = probabilities[class_idx] * 100
            
            # Feature 2: Dynamic Symptom Load Risk Tier Estimation Evaluation
            symptom_count = len(selected_clean)
            if symptom_count <= 2:
                risk_tier = "Low"
                risk_color = "#10b981"
            elif symptom_count <= 5:
                risk_tier = "Moderate"
                risk_color = "#f59e0b"
            else:
                risk_tier = "Elevated"
                risk_color = "#ef4444"

            # Display Key Metrics in side-by-side Columns
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Primary Target</div>
                        <div class="metric-value" style="color: #0284c7;">{prediction}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                    <div class="metric-box" style="border-left-color: #10b981;">
                        <div class="metric-label">Model Certainty</div>
                        <div class="metric-value" style="color: #10b981;">{confidence:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                    <div class="metric-box" style="border-left-color: {risk_color};">
                        <div class="metric-label">Symptom Load Tier</div>
                        <div class="metric-value" style="color: {risk_color};">{risk_tier}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Render Core Medical Information Panels
            if prediction in DISEASE_INFO:
                info = DISEASE_INFO[prediction]
                st.markdown(f"""
                    <div class="info-box">
                        <strong>Clinical Overview:</strong> {info['desc']}
                    </div>
                    <div class="info-box" style="border-left: 4px solid #f59e0b;">
                        📍 <strong>Recommended Unit Routing:</strong> Referral recommended to <strong>{info['specialist']}</strong>.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="info-box">
                        <strong>Clinical Overview:</strong> Conditional indicator tracking profiles pending expansion.
                    </div>
                """, unsafe_allow_html=True)

            # Feature 3: Alternate Probability Classifier Analytics Charts Layout
            st.markdown("<br><h5>Top Probable Variant Distributions</h5>", unsafe_allow_html=True)
            top_indices = np.argsort(probabilities)[::-1][:3]
            chart_data = pd.DataFrame({
                "Condition Class": [model.classes_[i] for i in top_indices],
                "Confidence Match Score (%)": [probabilities[i] * 100 for i in top_indices]
            })
            st.bar_chart(chart_data, x="Condition Class", y="Confidence Match Score (%)", color="#0284c7")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("⚠️ **Educational Project Disclaimer:** This system functions strictly as a data-science exercise using training datasets. It does not replace professional medical evaluations, clinical triage plans, or medical advice.")
