import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Clinical Diagnosis Portal",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD BACKGROUND IMAGE
# Put caregiver_bg.jpg in same folder as app.py
# =====================================================
def get_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        # Fallback to empty string if background image is missing
        return ""

img = get_base64("caregiver_bg.jpg")

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown(
    f"""
    <style>

    /* =====================================================
       MAIN BACKGROUND
    ===================================================== */
    .stApp {{
        background:
            linear-gradient(
                rgba(255,255,255,0.30),
                rgba(255,255,255,0.30)
            ),
            url("data:image/jpg;base64,{img}");

        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}

    .main .block-container {{
        background: transparent !important;
        padding-top: 2rem;
    }}

    /* =====================================================
       TYPOGRAPHY
    ===================================================== */
    h1 {{
        color: #172033 !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
    }}

    h2, h3, h4, h5 {{
        color: #172033 !important;
        font-weight: 700 !important;
    }}

    .sub-heading {{
        color: #334155 !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }}

    /* =====================================================
       HEADER CARD
    ===================================================== */
    .header-card {{
        background: rgba(255,255,255,0.82);
        backdrop-filter: blur(5px);

        padding: 30px;

        border-radius: 20px;

        border: 1px solid rgba(255,255,255,0.4);

        box-shadow: 0 8px 30px rgba(0,0,0,0.08);

        margin-bottom: 30px;
    }}

    /* =====================================================
       MAIN CARD
    ===================================================== */
    .main-card {{
        background: rgba(255,255,255,0.80);

        backdrop-filter: blur(5px);

        border-radius: 18px;

        padding: 25px;

        border: 1px solid rgba(255,255,255,0.35);

        box-shadow: 0 4px 20px rgba(0,0,0,0.06);

        margin-bottom: 25px;
    }}

    /* =====================================================
       SIDEBAR
    ===================================================== */
    section[data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.82) !important;

        backdrop-filter: blur(8px);
    }}

    /* =====================================================
       BUTTONS
    ===================================================== */
    .stButton > button {{
        width: 100%;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 18px !important;
        font-weight: 700 !important;
        transition: 0.3s;
    }}

    .stButton > button[data-testid="baseButton-primary"] {{
        background: #0f766e !important;
        color: white !important;
    }}

    .stButton > button[data-testid="baseButton-primary"]:hover {{
        background: #115e59 !important;
        transform: translateY(-2px);
    }}

    .stButton > button:not([data-testid="baseButton-primary"]) {{
        background: #e2e8f0 !important;
        color: #172033 !important;
    }}

    /* =====================================================
       METRIC CARDS
    ===================================================== */
    .clinical-metric {{
        background: rgba(255,255,255,0.82);

        backdrop-filter: blur(5px);

        border-radius: 16px;

        padding: 20px;

        border: 1px solid rgba(255,255,255,0.4);

        box-shadow: 0 4px 20px rgba(0,0,0,0.06);

        margin-top: 10px;
    }}

    .clinical-label {{
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}

    .clinical-value {{
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: 5px;
        color: #172033;
    }}

    /* =====================================================
       INFO BOXES
    ===================================================== */
    .clinical-info-bin {{
        background: rgba(255,255,255,0.82);

        backdrop-filter: blur(5px);

        border-radius: 16px;

        padding: 18px;

        border: 1px solid rgba(255,255,255,0.35);

        box-shadow: 0 4px 18px rgba(0,0,0,0.05);

        margin-top: 15px;

        color: #334155;

        line-height: 1.6;
    }}

    /* =====================================================
       DATAFRAME
    ===================================================== */
    [data-testid="stDataFrame"] {{
        background: rgba(255,255,255,0.82);
        border-radius: 15px;
        overflow: hidden;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <div class="header-card">

        <h2 style="color:#0f766e; margin-bottom:8px;">
            🏥 Clinical Informatics Intelligence Center
        </h2>

        <p style="color:#475569; margin:0; font-size:1rem;">
            Machine Learning Statistical Inference Gateway
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL
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

    # Sanitize feature names for presentation (replace underscores)
    feature_mapping = {f: f.replace('_', ' ').title() for f in X.columns}
    return model, X.columns, feature_mapping

try:
    model, raw_features, clean_feature_map = load_and_train_model()
    model_ready = True
except Exception:
    st.error("Training.csv file not found or corrupted.")
    model_ready = False

# =====================================================
# DISEASE DATABASE
# =====================================================
DISEASE_INFO = {
    "Fungal infection": {
        "desc": "A fungal skin infection affecting warm and moist areas.",
        "specialist": "Dermatologist",
        "treatment": "Keep skin dry and apply antifungal cream."
    },
    "Allergy": {
        "desc": "Immune system reaction caused by allergens.",
        "specialist": "Allergist",
        "treatment": "Avoid allergens and use antihistamines."
    },
    "Diabetes ": {
        "desc": "A disease causing high blood sugar levels.",
        "specialist": "Endocrinologist",
        "treatment": "Monitor blood sugar and follow proper diet."
    },
    "Migraine": {
        "desc": "A neurological condition causing severe headaches.",
        "specialist": "Neurologist",
        "treatment": "Rest in dark rooms and avoid stress."
    },
    "Hypertension ": {
        "desc": "High blood pressure condition.",
        "specialist": "Cardiologist",
        "treatment": "Reduce salt intake and monitor BP regularly."
    }
}

# =====================================================
# SESSION STATE
# =====================================================
if "history_log" not in st.session_state:
    st.session_state.history_log = []

# =====================================================
# MAIN USER INTERFACE
# =====================================================
st.title("AI Clinical Diagnosis Portal")
st.markdown('<p class="sub-heading">Isolate specific patient symptom indicators to compute machine learning prediction pathways.</p>', unsafe_allow_html=True)

if model_ready:
    # Sidebar Configuration for inputs
    st.sidebar.header("Patient Profiles & Symptoms")
    
    # Allow mapping presentation titles back to original dataframe column keys
    reverse_feature_map = {v: k for k, v in clean_feature_map.items()}
    display_options = sorted(list(clean_feature_map.values()))
    
    selected_clean_symptoms = st.sidebar.multiselect(
        "Select Manifested Symptoms:",
        options=display_options,
        help="Choose all symptoms currently expressed by the individual."
    )

    # Actions Container
    st.sidebar.markdown("---")
    compute_btn = st.sidebar.button("Compute Analysis Pathway", type="primary")
    clear_btn = st.sidebar.button("Clear Diagnostic History")

    if clear_btn:
        st.session_state.history_log = []
        st.rerun()

    # Main Workflow Engine
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Diagnostic Engine Evaluation")
    
    if compute_btn:
        if not selected_clean_symptoms:
            st.warning("Please choose one or more symptoms from the sidebar configuration panel before requesting a prediction.")
        else:
            # Build binary vector mapping to dataset columns
            input_vector = np.zeros(len(raw_features))
            for sym in selected_clean_symptoms:
                raw_key = reverse_feature_map[sym]
                idx = list(raw_features).index(raw_key)
                input_vector[idx] = 1

            # Execute Statistical Inference
            with st.spinner("Processing network weights..."):
                time.sleep(0.6)  # UI feedback delay
                prediction = model.predict([input_vector])[0]
                probabilities = model.predict_proba([input_vector])[0]
                
                class_idx = list(model.classes_).index(prediction)
                confidence = probabilities[class_idx] * 100

            # Log execution events into history registry
            timestamp = time.strftime("%H:%M:%S")
            st.session_state.history_log.insert(0, {
                "Time": timestamp,
                "Identified Classification": prediction,
                "Confidence Interval": f"{confidence:.1f}%",
                "Symptom Count": len(selected_clean_symptoms)
            })

            # Layout metrics side-by-side using card syntax
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

            # Extract detailed profiles if available in database mapping
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
                st.markdown("""
                <div class="clinical-info-bin">
                    <strong>Pathology Profile:</strong> Custom clinical profile undefined in base database maps. Consult external health references.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("System idling. Configure symptoms on the left sidebar framework and execute computation.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # History Data Grid Logs
    if st.session_state.history_log:
        st.markdown("### Real-Time Session Execution Logs")
        log_df = pd.DataFrame(st.session_state.history_log)
        st.dataframe(log_df, use_container_width=True)
