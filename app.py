import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Clinical Diagnostic Portal",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD BACKGROUND IMAGE
# Put caregiver_bg.jpg in same folder as app.py
# =========================================================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("caregiver_bg.jpg")

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(f"""
<style>

/* =========================================================
BACKGROUND IMAGE
========================================================= */
.stApp {{
    background:
        linear-gradient(
            rgba(255,255,255,0.30),
            rgba(255,255,255,0.30)
        ),
        url("data:image/jpg;base64,{img}");

    background-size: cover !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
}}

/* REMOVE STREAMLIT WHITE LAYER */
[data-testid="stAppViewContainer"] {{
    background: transparent !important;
}}

.main .block-container {{
    background: transparent !important;
    padding-top: 2rem;
}}

/* =========================================================
TYPOGRAPHY
========================================================= */
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

/* =========================================================
HEADER CARD
========================================================= */
.header-card {{
    background: rgba(255,255,255,0.80);
    backdrop-filter: blur(6px);

    padding: 30px;
    border-radius: 20px;

    border: 1px solid rgba(255,255,255,0.4);

    box-shadow: 0 8px 30px rgba(0,0,0,0.08);

    margin-bottom: 30px;
}}

/* =========================================================
MAIN CARD
========================================================= */
.main-card {{
    background: rgba(255,255,255,0.78);

    backdrop-filter: blur(5px);

    border-radius: 20px;

    padding: 25px;

    border: 1px solid rgba(255,255,255,0.35);

    box-shadow: 0 4px 20px rgba(0,0,0,0.06);

    margin-bottom: 25px;
}}

/* =========================================================
SIDEBAR
========================================================= */
section[data-testid="stSidebar"] {{
    background: rgba(255,255,255,0.82) !important;

    backdrop-filter: blur(8px);
}}

/* =========================================================
BUTTONS
========================================================= */
.stButton > button {{
    width: 100%;
    border-radius: 12px !important;
    border: none !important;
    padding: 12px 20px !important;
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

/* =========================================================
METRIC CARDS
========================================================= */
.clinical-metric {{
    background: rgba(255,255,255,0.80);

    backdrop-filter: blur(6px);

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

/* =========================================================
INFO BOXES
========================================================= */
.clinical-info-bin {{
    background: rgba(255,255,255,0.80);

    backdrop-filter: blur(5px);

    border-radius: 16px;

    padding: 18px;

    border: 1px solid rgba(255,255,255,0.35);

    box-shadow: 0 4px 18px rgba(0,0,0,0.05);

    margin-top: 15px;

    color: #334155;

    line-height: 1.6;
}}

/* =========================================================
DATAFRAME
========================================================= */
[data-testid="stDataFrame"] {{
    background: rgba(255,255,255,0.82);
    border-radius: 15px;
    overflow: hidden;
}}

/* =========================================================
MULTISELECT
========================================================= */
.stMultiSelect,
.stSelectbox {{
    background: rgba(255,255,255,0.7);
    border-radius: 10px;
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="header-card">

    <h2 style="color:#0f766e; margin-bottom:8px;">
        🏥 Clinical Informatics Intelligence Center
    </h2>

    <p style="color:#475569; margin:0; font-size:1rem;">
        Machine Learning Statistical Inference Gateway
    </p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
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

except Exception:
    st.error("Training.csv file not found.")
    model_ready = False

# =========================================================
# DISEASE INFO
# =========================================================
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

# =========================================================
# SESSION STATE
# =========================================================
if "history_log" not in st.session_state:
    st.session_state.history_log = []

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Patient Entry Profiles")

patient_age = st.sidebar.slider(
    "Patient Age",
    min_value=1,
    max_value=100,
    value=30
)

patient_gender = st.sidebar.selectbox(
    "Biological Sex",
    ["Male", "Female", "Other"]
)

st.sidebar.markdown("---")

symptom_duration = st.sidebar.selectbox(
    "Symptoms Persisting For",
    [
        "Less than 24 Hours",
        "1 to 3 Days",
        "More than a Week"
    ]
)

# =========================================================
# MAIN TITLE
# =========================================================
st.title("AI Clinical Diagnosis Portal")

st.markdown("""
<div class="sub-heading">
Isolate specific patient symptom indicators to compute machine learning prediction pathways.
</div>
""", unsafe_allow_html=True)

# =========================================================
# MAIN APP
# =========================================================
if model_ready:

    clean_features = [
        f.replace("_", " ").title()
        for f in features
    ]

    if "symptom_key" not in st.session_state:
        st.session_state.symptom_key = 0

    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    selected_clean = st.multiselect(
        "Identify Patient Manifestations",
        clean_features,
        placeholder="Select symptoms...",
        key=f"symptoms_{st.session_state.symptom_key}"
    )

    col1, col2 = st.columns(2)

    with col1:
        submit_btn = st.button(
            "Run Diagnostic Analysis",
            type="primary"
        )

    with col2:
        if st.button("Clear Input"):
            st.session_state.symptom_key += 1
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================
    # PREDICTION
    # =========================================================
    if submit_btn:

        if not selected_clean:

            st.warning("Please select symptoms.")

        else:

            start_time = time.perf_counter()

            input_matrix = np.zeros((1, len(features)))

            for clean_sym in selected_clean:

                raw_name = clean_sym.lower().replace(" ", "_")

                if raw_name in features:

                    idx = features.index(raw_name)

                    input_matrix[0, idx] = 1

            prediction_raw = model.predict(input_matrix)

            prediction = str(prediction_raw[0]).strip()

            probabilities = model.predict_proba(input_matrix).flatten()

            classes = [str(c).strip() for c in model.classes_]

            if prediction in classes:

                class_idx = classes.index(prediction)

                confidence = probabilities[class_idx] * 100

            else:
                confidence = 0

            latency = (time.perf_counter() - start_time) * 1000

            # =========================================================
            # RISK
            # =========================================================
            symptom_count = len(selected_clean)

            if symptom_count <= 2:

                risk = "Mild"
                risk_color = "#0f766e"

            elif symptom_count <= 5:

                risk = "Moderate"
                risk_color = "#d97706"

            else:

                risk = "High Risk"
                risk_color = "#dc2626"

            # =========================================================
            # SAVE HISTORY
            # =========================================================
            st.session_state.history_log.append({
                "Condition": prediction,
                "Confidence": f"{confidence:.1f}%",
                "Risk": risk
            })

            # =========================================================
            # METRICS
            # =========================================================
            m1, m2, m3 = st.columns(3)

            with m1:

                st.markdown(f"""
                <div class="clinical-metric">

                    <div class="clinical-label">
                        Diagnosis
                    </div>

                    <div class="clinical-value" style="color:#0f766e;">
                        {prediction}
                    </div>

                </div>
                """, unsafe_allow_html=True)

            with m2:

                st.markdown(f"""
                <div class="clinical-metric">

                    <div class="clinical-label">
                        Confidence
                    </div>

                    <div class="clinical-value">
                        {confidence:.1f}%
                    </div>

                </div>
                """, unsafe_allow_html=True)

            with m3:

                st.markdown(f"""
                <div class="clinical-metric">

                    <div class="clinical-label">
                        Risk Level
                    </div>

                    <div class="clinical-value" style="color:{risk_color};">
                        {risk}
                    </div>

                </div>
                """, unsafe_allow_html=True)

            st.markdown(
                f"""
                <p style="
                    text-align:right;
                    color:#334155;
                    font-size:0.85rem;
                    margin-top:10px;
                    font-weight:600;
                ">
                    Inference Latency: {latency:.3f} ms
                </p>
                """,
                unsafe_allow_html=True
            )

            # =========================================================
            # DISEASE INFO
            # =========================================================
            desc = "No information available."
            specialist = "General Physician"
            treatment = "Consult a doctor."

            if prediction in DISEASE_INFO:

                desc = DISEASE_INFO[prediction]["desc"]
                specialist = DISEASE_INFO[prediction]["specialist"]
                treatment = DISEASE_INFO[prediction]["treatment"]

            st.markdown(f"""
            <div class="clinical-info-bin">

                <strong>Medical Overview:</strong>

                <br><br>

                {desc}

            </div>

            <div class="clinical-info-bin">

                👨‍⚕️ <strong>Recommended Specialist:</strong>

                {specialist}

            </div>

            <div class="clinical-info-bin">

                🛡️ <strong>Suggested Treatment:</strong>

                <br><br>

                {treatment}

            </div>
            """, unsafe_allow_html=True)

            # =========================================================
            # CHART
            # =========================================================
            st.markdown("### Prediction Statistics")

            top_indices = np.argsort(probabilities)[::-1][:3]

            chart_data = pd.DataFrame({
                "Condition": [classes[i] for i in top_indices],
                "Confidence (%)": [
                    probabilities[i] * 100
                    for i in top_indices
                ]
            })

            st.bar_chart(
                chart_data,
                x="Condition",
                y="Confidence (%)"
            )

            # =========================================================
            # OTHER PREDICTIONS
            # =========================================================
            alt_indices = np.argsort(probabilities)[::-1][1:6]

            matrix_df = pd.DataFrame({
                "Alternative Predictions": [
                    classes[i]
                    for i in alt_indices
                ],
                "Probability": [
                    f"{probabilities[i] * 100:.2f}%"
                    for i in alt_indices
                ]
            })

            st.dataframe(
                matrix_df,
                use_container_width=True,
                hide_index=True
            )

            # =========================================================
            # REPORT DOWNLOAD
            # =========================================================
            report_content = f"""
AI CLINICAL REPORT

Patient Age: {patient_age}
Gender: {patient_gender}

Symptoms:
{", ".join(selected_clean)}

Prediction:
{prediction}

Confidence:
{confidence:.2f}%

Risk:
{risk}

Specialist:
{specialist}

Treatment:
{treatment}
"""

            st.download_button(
                label="📥 Export Report",
                data=report_content,
                file_name="Clinical_Report.txt",
                mime="text/plain",
                use_container_width=True
            )

# =========================================================
# HISTORY
# =========================================================
if st.session_state.history_log:

    st.markdown("## 📜 Session History")

    history_df = pd.DataFrame(
        st.session_state.history_log[::-1]
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("<br><br>", unsafe_allow_html=True)

st.caption("""
⚠️ Educational Project Disclaimer:
This system is for educational purposes only and does not replace professional medical diagnosis.
""")
