import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

# Set a professional dashboard page layout
st.set_page_config(
    page_title="Advanced AI Medical Diagnostics Node",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Premium Clinical Sky Blue & Clean Slate CSS Styling
st.markdown("""
    <style>
    /* Clean Solid White App Background */
    .stApp {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Clean Deep Slate Typography Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* Clear Subtitle Description Text styling */
    .sub-heading {
        color: #475569 !important;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Sidebar Styling Configuration (Soft Ice Blue tint) */
    section[data-testid="stSidebar"] {
        background-color: #f0f7ff !important;
        border-right: 1px solid #bae6fd;
    }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {
        color: #0369a1 !important;
        font-weight: 600;
    }

    /* Core Symptom Form White Block Container Box */
    div[data-testid="stVerticalBlock"] > div:has(div.stMultiSelect) {
        background: #ffffff !important;
        padding: 35px !important;
        border-radius: 12px !important;
        border: 1px solid #e0f2fe !important;
        box-shadow: 0 4px 20px -2px rgba(14, 165, 233, 0.08) !important;
    }

    /* Premium Sky Blue Primary Action Button Styling */
    .stButton>button[data-testid="baseButton-primary"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2) !important;
        transition: all 0.2s ease;
    }
    .stButton>button[data-testid="baseButton-primary"]:hover {
        background-color: #0369a1 !important;
        box-shadow: 0 4px 16px rgba(3, 105, 161, 0.3) !important;
    }
    
    /* Sky Blue Clinical Output Display Metric Cards */
    .clinical-metric {
        background-color: #f0f9ff;
        border: 1px solid #e0f2fe;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        border-top: 4px solid #0284c7;
    }
    .clinical-label {
        font-size: 0.75rem;
        color: #0369a1;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .clinical-value {
        font-size: 1.25rem;
        color: #0c4a6e;
        font-weight: 700;
        margin-top: 4px;
    }
    
    /* Soft Blue Informational Output Alert Blocks */
    .clinical-info-bin {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 16px;
        border-radius: 8px;
        margin-top: 12px;
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.6;
        border-left: 4px solid #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

# CSS-based Corporate Medical Header Box (Deep Sky Navy Slate variant)
st.markdown("""
    <div style="background-color: #0c4a6e; padding: 30px; border-radius: 8px; text-align: left; margin-bottom: 25px; border-bottom: 4px solid #38bdf8;">
        <h2 style="color: #ffffff; margin: 0; font-family: 'Inter', sans-serif; font-size: 1.5rem; font-weight: 700;">🏥 Clinical Informatics Intelligence Center</h2>
        <p style="color: #e0f2fe; margin: 6px 0 0 0; font-size: 0.9rem; font-weight: 400; letter-spacing: 0.5px;">Machine Learning Statistical Inference Gateway</p>
    </div>
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

# Mapping database tracking for predictions with treatment engine updates
DISEASE_INFO = {
    "Fungal infection": {"desc": "A skin condition caused by fungus targeting warm, moist body areas.", "specialist": "Dermatologist", "treatment": "Keep skin dry, apply topical over-the-counter antifungals, avoid scratching."},
    "Allergy": {"desc": "An immune system reaction to a foreign substance or allergen exposure.", "specialist": "Allergist / Immunologist", "treatment": "Identify and avoid allergen triggers, utilize non-drowsy antihistamines."},
    "GERD": {"desc": "Gastroesophageal reflux disease causes digestive acids to irritate the food pipe.", "specialist": "Gastroenterologist", "treatment": "Avoid lying down immediately after eating, minimize acidic foods, consume smaller meals."},
    "Chronic cholestasis": {"desc": "A condition where bile flow from the liver is reduced or stalled.", "specialist": "Hepatologist / Gastroenterologist", "treatment": "Consult specialist for medication review, monitor liver lipid profile markers."},
    "Drug Reaction": {"desc": "An adverse side effect triggered by an ingested or applied medication.", "specialist": "General Physician / Allergist", "treatment": "Cease suspected secondary medication instantly, contact prescribing professional."},
    "Peptic ulcer diseae": {"desc": "Sores that develop on the inner lining of your stomach and upper small intestine.", "specialist": "Gastroenterologist", "treatment": "Avoid NSAID pain relievers, eliminate spicy foods, track lifestyle indicators closely."},
    "AIDS": {"desc": "A chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV).", "specialist": "Infectious Disease Specialist", "treatment": "Strict compliance with prescribed Antiretroviral Therapy (ART), routine viral tracking."},
    "Diabetes ": {"desc": "A group of diseases that result in too much sugar in the blood (high blood glucose).", "specialist": "Endocrinologist", "treatment": "Monitor blood glucose tracking parameters, adopt consistent dietary regimens."},
    "Gastroenteritis": {"desc": "An intestinal infection marked by diarrhea, cramps, nausea, vomiting, and fever.", "specialist": "Gastroenterologist", "treatment": "Prioritize oral rehydration solutions (ORS), maintain a plain low-fiber diet plan."},
    "Bronchial Asthma": {"desc": "A condition in which a person's airways become inflamed, narrow and swell, and produce extra mucus.", "specialist": "Pulmonologist / Allergist", "treatment": "Keep rescue inhaler at hand, monitor local environmental particle triggers."},
    "Hypertension ": {"desc": "A condition in which the force of the blood against the artery walls is too high.", "specialist": "Cardiologist", "treatment": "Implement low-sodium nutritional limits, establish routine blood pressure checks."},
    "Migraine": {"desc": "A headache of varying intensity, often accompanied by nausea and sensitivity to light and sound.", "specialist": "Neurologist", "treatment": "Rest in dark rooms during attacks, avoid excessive screen times and processed tracking triggers."},
    "Cervical spondylosis": {"desc": "Age-related wear and tear affecting the spinal disks in your neck.", "specialist": "Orthopedic Surgeon / Neurologist", "treatment": "Apply warm compresses, perform gentle neck mobility stretching routines daily."},
    "Paralysis (brain hemorrhage)": {"desc": "Loss of muscle function in part of your body, caused by bleeding inside the brain tissue.", "specialist": "Neurologist / Neurosurgeon", "treatment": "Immediate emergency room evaluations required, establish supportive physical therapies."},
    "Jaundice": {"desc": "A yellowing of the skin and eyes caused by high levels of bilirubin.", "specialist": "Hepatologist", "treatment": "Rest properly, maintain low-fat hydration fluid intakes, follow targeted medical plans."}
}

if model_ready:
    st.sidebar.header("Diagnostics Control Panel")
    st.sidebar.write("Select patient features to begin inference analysis.")
    
    # Format symptom strings for clean display
    display_features = [f.replace("_", " ").title() for f in features]
    feature_mapping = dict(zip(display_features, features))
    
    # Input selection UI element
    selected_display = st.multiselect(
        "Select Present Symptoms:", 
        options=display_features,
        help="Type or select multiple symptoms from the medical ledger index."
    )
    
    # Action submission logic execution
    if st.button("Run Diagnostic Inference", type="primary"):
        if not selected_display:
            st.warning("Please choose at least one present symptom before calculating outcomes.")
        else:
            with st.spinner("Processing clinical data arrays..."):
                time.sleep(1.2) # Artificial simulation for premium UX feel
                
                # Setup input vector matrix
                input_vector = np.zeros(len(features))
                for sd in selected_display:
                    feature_raw_name = feature_mapping[sd]
                    idx = features.index(feature_raw_name)
                    input_vector[idx] = 1
                
                # Fetch classifier results
                prediction = model.predict([input_vector])[0]
                probabilities = model.predict_proba([input_vector])[0]
                confidence = np.max(probabilities) * 100
                
                st.subheader("Analysis Insights & Results")
                
                # Render clean metric components using injected CSS styles
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div class="clinical-metric">
                            <div class="clinical-label">Primary Prognosis Inference</div>
                            <div class="clinical-value">{prediction}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="clinical-metric">
                            <div class="clinical-label">Statistical Confidence</div>
                            <div class="clinical-value">{confidence:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Display mapped clinical descriptions safely
                info_node = DISEASE_INFO.get(prediction, {"desc": "No specific descriptive matrix logged.", "specialist": "General Physician", "treatment": "Consult standard care path directions."})
                
                st.markdown(f"""
                    <div class="clinical-info-bin">
                        <strong>Pathology Profile:</strong> {info_node['desc']}<br><br>
                        <strong>Recommended Consultant Pathway:</strong> {info_node['specialist']}<br><br>
                        <strong>Standard Support Strategy:</strong> {info_node['treatment']}
                    </div>
                """, unsafe_allow_html=True)
