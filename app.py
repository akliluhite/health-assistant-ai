import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Clinical Diagnostic Portal",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Medical Care Background Image Link URL mapping
BACKGROUND_IMAGE_URL = "https://githubusercontent.com"

# =====================================================
# PREMIUM MATERIAL UI CLINICAL CSS SYSTEM
# =====================================================
st.markdown(f"""
    <style>
    /* Full-Screen Structural Background Image configurations */
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), url("{BACKGROUND_IMAGE_URL}");
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        color: #334155 !important;
    }}
    
    /* Modern Slate Primary Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: #1e293b !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 600 !important;
        letter-spacing: -0.25px;
    }}
    
    /* Sub-heading typography normalization */
    .sub-heading {{
        color: #475569 !important;
        font-size: 1rem;
        margin-bottom: 2rem;
        line-height: 1.5;
        font-weight: 500;
    }}
    
    /* Slate Grey Minimalist Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: rgba(248, 250, 252, 0.95) !important;
        border-right: 1px solid #e2e8f0;
    }}
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {{
        color: #475569 !important;
        font-weight: 500;
    }}

    /* Semi-Transparent Glassmorphism Workspace Content Container Box */
    div[data-testid="stVerticalBlock"] > div:has(div.stMultiSelect) {{
        background: rgba(255, 255, 255, 0.95) !important;
        padding: 30px !important;
        border-radius: 12px !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        backdrop-filter: blur(8px);
    }}

    /* Flat Teal Interactive Action Trigger Button Styling */
    .stButton>button[data-testid="baseButton-primary"] {{
        background-color: #0f766e !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        transition: background-color 0.15s ease;
        width: 100%;
    }}
    .stButton>button[data-testid="baseButton-primary"]:hover {{
        background-color: #115e59 !important;
    }}
    
    .stButton > button:not([data-testid="baseButton-primary"]) {{
        background: #e2e8f0 !important;
        color: #172033 !important;
        border-radius: 6px !important;
        border: none !important;
        width: 100%;
    }}
    
    /* Clean White Metric Component Blocks */
    .clinical-metric {{
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 18px;
        margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }}
    .clinical-label {{
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .clinical-value {{
        font-size: 1.2rem;
        color: #1e293b;
        font-weight: 600;
        margin-top: 4px;
    }}
    
    /* General Flat Content Information Bins */
    .clinical-info-bin {{
        background-color: rgba(248, 250, 252, 0.95);
        border: 1px solid #cbd5e1;
        padding: 16px;
        border-radius: 8px;
        margin-top: 12px;
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    
    [data-testid="stDataFrame"] {{
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #cbd5e1;
    }}
    </style>
""", unsafe_allow_html=True)

# =====================================================
# CORPORATE MEDICAL HEADER CARD
# =====================================================
st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.9); padding: 30px; border-radius: 12px; text-align: left; margin-bottom: 25px; border: 1px solid #cbd5e1; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);">
        <h2 style="color: #0f766e; margin: 0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.5rem; font-weight: 700;">🏥 Clinical Informatics Intelligence Center</h2>
        <div style="color: #475569; margin-top: 8px; font-size: 0.95rem; font-weight: 400; letter-spacing: 0.5px;">Machine Learning Statistical Inference Gateway</div>
    </div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD AND TRAIN DATA MODEL SYSTEM
# =====================================================
@st.cache_resource
def load_and_train_model():
    data = pd.read_csv("Training.csv")
    if "Unnamed: 133" in data.columns:
        data = data.drop(columns=["Unnamed: 133"])
    X = data.drop(columns=["prognosis"])
    y = data["prognosis"]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X.values, y)
    
    # Generate human-readable labels from under_scored feature columns
    feature_mapping = {f: f.replace('_', ' ').title() for f in X.columns}
    return model, X.columns, feature_mapping

try:
    model, raw_features, clean_feature_map = load_and_train_model()
    model_ready = True
except Exception as e:
    st.error("Could not load 'Training.csv' locally. Please verify your file setup.")
    model_ready = False

# =====================================================
# CLINICAL PATHOLOGY PROFILES REGISTER
# =====================================================
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
    "Migraine": {"desc": "A headache of varying intensity, often accompanied by nausea and sensitivity to light and sound.", "specialist": "Neurologist", "treatment": "Rest in dark rooms during attacks, avoid excessive screen strain or triggers."}
}

# =====================================================
# SESSION TRACKING LOGS
# =====================================================
if "history_log" not in st.session_state:
    st.session_state.history_log = []

# =====================================================
# MAIN FRAMEWORK BODY
# =====================================================
st.title("AI Clinical Diagnosis Portal")
st.markdown('<p class="sub-heading">Isolate specific patient symptom indicators to compute machine learning prediction pathways.</p>', unsafe_allow_html=True)

if model_ready:
    # Sidebar control modules
    st.sidebar.header("Patient Profiles & Symptoms")
    
    # Reverse mapping helper to convert selection back to raw database indices
    reverse_feature_map = {v: k for k, v in clean_feature_map.items()}
    display_options = sorted(list(clean_feature_map.values()))
    
    selected_clean_symptoms = st.sidebar.multiselect(
        "Select Manifested Symptoms:",
        options=display_options,
        help="Choose all physical indicators reported by the target patient."
    )

    st.sidebar.markdown("---")
    compute_btn = st.sidebar.button("Compute Analysis Pathway", type="primary")
    clear_btn = st.sidebar.button("Clear Diagnostic History")

    if clear_btn:
        st.session_state.history_log = []
        st.rerun()

    # Diagnostic workspace dashboard card
    st.subheader("Diagnostic Engine Evaluation")
    
    if compute_btn:
        if not selected_clean_symptoms:
            st.warning("Please choose one or more symptoms from the sidebar configuration panel before requesting a prediction.")
        else:
            # Match strings back into structured array matrices
            input_vector = np.zeros(len(raw_features))
            for sym in selected_clean_symptoms:
                raw_key = reverse_feature_map[sym]
                idx = list(raw_features).index(raw_key)
                input_vector[idx] = 1

            # Engine Prediction Execution Blocks
            with st.spinner("Processing structural node paths..."):
                time.sleep(0.5)
                prediction = model.predict([input_vector])[0]
                probabilities = model.predict_proba([input_vector])
                class_idx = list(model.classes_).index(prediction)
                confidence = probabilities[class_idx] * 100

            # Store tracking information to user execution registry state
            timestamp = time.strftime("%H:%M:%S")
            st.session_state.history_log.insert(0, {
                "Time": timestamp,
                "Identified Classification": prediction,
                "Confidence Interval": f"{confidence:.1f}%",
                "Symptom Indicators": len(selected_clean_symptoms)
            })

            # Present Metrics via Cards
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class="clinical-metric">
                        <div class="clinical-label">Inferred Classification Prognosis</div>
                        <div class="clinical-value">{prediction}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="clinical-metric">
                        <div class="clinical-label">Statistical Weight Confidence</div>
                        <div class="clinical-value">{confidence:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

            # Match and show deep clinical profile specs
            disease_profile = DISEASE_INFO.get(prediction, DISEASE_INFO.get(prediction.strip()))
            if disease_profile:
                st.markdown(f"""
                    <div class="clinical-info-bin">
                        <strong>Pathology Profile:</strong> {disease_profile['desc']}<br><br>
                        <strong>Target Department Referral:</strong> {disease_profile['specialist']}<br>
                        <strong>Recommended Actions:</strong> {disease_profile['treatment']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="clinical-info-bin">
                        <strong>Pathology Profile:</strong> Custom classification matches external reference data. Verify specific biomarkers with secondary tests.
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("System idling. Configure clinical profile conditions on the panel to compile inferences.")

    # Render History DataFrames dynamically
    if st.session_state.history_log:
        st.markdown("### Real-Time Session Execution Logs")
        log_df = pd.DataFrame(st.session_state.history_log)
        st.dataframe(log_df, use_container_width=True)
