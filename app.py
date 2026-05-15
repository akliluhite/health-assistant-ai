import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
from sklearn.ensemble import RandomForestClassifier

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="AI Clinical Diagnostic Portal",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =========================
# BACKGROUND IMAGE FUNCTION
# =========================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Put caregiver_bg.jpg in same folder as app.py
img = get_base64("caregiver_bg.jpg")

# =========================
# PREMIUM UI CSS
# =========================
st.markdown(f"""
<style>

/* =========================
   MAIN APP BACKGROUND
========================= */
.stApp {{
    background:
        linear-gradient(
            rgba(255,255,255,0.82),
            rgba(255,255,255,0.82)
        ),
        url("data:image/jpg;base64,{img}");

    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;

    color: #334155 !important;
}}

/* =========================
   TYPOGRAPHY
========================= */
h1, h2, h3, h4, h5, h6 {{
    color: #1e293b !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-weight: 700 !important;
    letter-spacing: -0.25px;
}}

.sub-heading {{
    color: #475569 !important;
    font-size: 1rem;
    margin-bottom: 2rem;
    line-height: 1.5;
    font-weight: 500;
}}

/* =========================
   SIDEBAR
========================= */
section[data-testid="stSidebar"] {{
    background: rgba(255,255,255,0.88) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid #e2e8f0;
}}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div {{
    color: #334155 !important;
}}

/* =========================
   MAIN CONTAINER
========================= */
.main-card {{
    background: rgba(255,255,255,0.90);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}}

/* =========================
   BUTTONS
========================= */
.stButton>button {{
    width: 100%;
    border-radius: 10px !important;
    border: none !important;
    padding: 12px 18px !important;
    font-weight: 600 !important;
    transition: 0.3s;
}}

.stButton>button[data-testid="baseButton-primary"] {{
    background: #0f766e !important;
    color: white !important;
}}

.stButton>button[data-testid="baseButton-primary"]:hover {{
    background: #115e59 !important;
    transform: translateY(-2px);
}}

.stButton>button:not([data-testid="baseButton-primary"]) {{
    background: #e2e8f0 !important;
    color: #1e293b !important;
}}

/* =========================
   METRIC CARDS
========================= */
.clinical-metric {{
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(8px);
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.5);
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin-top: 15px;
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
    color: #1e293b;
    font-weight: 700;
    margin-top: 6px;
}}

/* =========================
   INFO BOXES
========================= */
.clinical-info-bin {{
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(8px);
    border-radius: 12px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.5);
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    margin-top: 14px;
    color: #334155;
    line-height: 1.6;
}}

/* =========================
   HEADER CARD
========================= */
.header-card {{
    background: rgba(255,255,255,0.90);
    backdrop-filter: blur(10px);
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.5);
    box-shadow: 0 8px 30px rgba(0,0,0,0.06);
}}

/* =========================
   DATAFRAME
========================= */
[data-testid="stDataFrame"] {{
    background: rgba(255,255,255,0.88);
    border-radius: 12px;
    overflow: hidden;
}}

/* =========================
   SELECT BOX & MULTISELECT
========================= */
.stMultiSelect,
.stSelectbox {{
    background: rgba(255,255,255,0.85);
    border-radius: 10px;
}}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header-card">
    <h2 style="color:#0f766e; margin-bottom:8px;">
        🏥 Clinical Informatics Intelligence Center
    </h2>

    <p style="color:#475569; margin:0;">
        Machine Learning Statistical Inference Gateway
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# MODEL LOADING
# =========================
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
    st.error("Training.csv not found.")
    model_ready = False

# =========================
# SIMPLE DISEASE DATABASE
# =========================
DISEASE_INFO = {
    "Fungal infection": {
        "desc": "A skin condition caused by fungus.",
        "specialist": "Dermatologist",
        "treatment": "Keep skin dry and use antifungal cream."
    },

    "Allergy": {
        "desc": "Immune reaction caused by allergens.",
        "specialist": "Allergist",
        "treatment": "Avoid allergens and use antihistamines."
    },

    "Diabetes ": {
        "desc": "High blood sugar condition.",
        "specialist": "Endocrinologist",
        "treatment": "Monitor blood sugar and follow diet control."
    },

    "Migraine": {
        "desc": "Neurological headache condition.",
        "specialist": "Neurologist",
        "treatment": "Rest in dark rooms and avoid stress."
    },

    "Hypertension ": {
        "desc": "High blood pressure condition.",
        "specialist": "Cardiologist",
        "treatment": "Reduce salt intake and monitor BP regularly."
    }
}

# =========================
# SESSION STATE
# =========================
if "history_log" not in st.session_state:
    st.session_state.history_log = []

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Patient Entry Profiles")

patient_age = st.sidebar.slider(
    "Patient Age",
    1,
    100,
    30
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

# =========================
# MAIN TITLE
# =========================
st.title("AI Clinical Diagnosis Portal")

st.markdown("""
<div class='sub-heading'>
Isolate specific patient symptom indicators to compute machine learning prediction pathways.
</div>
""", unsafe_allow_html=True)

# =========================
# MAIN CONTENT
# =========================
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
        if st.button("Clear Input Node"):
            st.session_state.symptom_key += 1
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # PREDICTION
    # =========================
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

            # =========================
            # RISK LEVEL
            # =========================
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

            # =========================
            # HISTORY
            # =========================
            st.session_state.history_log.append({
                "Condition": prediction,
                "Confidence": f"{confidence:.1f}%",
                "Risk": risk
            })

            # =========================
            # METRICS
            # =========================
            m1, m2, m3 = st.columns(3)

            with m1:
                st.markdown(f"""
                <div class="clinical-metric">
                    <div class="clinical-label">Diagnosis</div>
                    <div class="clinical-value" style="color:#0f766e;">
                        {prediction}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="clinical-metric">
                    <div class="clinical-label">Confidence</div>
                    <div class="clinical-value">
                        {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                st.markdown(f"""
                <div class="clinical-metric">
                    <div class="clinical-label">Risk Level</div>
                    <div class="clinical-value" style="color:{risk_color};">
                        {risk}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(
                f"""
                <p style='text-align:right;
                          color:#475569;
                          font-size:0.8rem;
                          margin-top:10px;'>
                    Inference Latency: {latency:.3f} ms
                </p>
                """,
                unsafe_allow_html=True
            )

            # =========================
            # DISEASE INFO
            # =========================
            desc = "No description available."
            specialist = "General Physician"
            treatment = "Consult doctor."

            if prediction in DISEASE_INFO:

                desc = DISEASE_INFO[prediction]["desc"]
                specialist = DISEASE_INFO[prediction]["specialist"]
                treatment = DISEASE_INFO[prediction]["treatment"]

            st.markdown(f"""
            <div class="clinical-info-bin">
                <strong>Medical Overview:</strong><br><br>
                {desc}
            </div>

            <div class="clinical-info-bin">
                👨‍⚕️ <strong>Recommended Specialist:</strong>
                {specialist}
            </div>

            <div class="clinical-info-bin">
                🛡️ <strong>Suggested First Aid / Care:</strong><br><br>
                {treatment}
            </div>
            """, unsafe_allow_html=True)

            # =========================
            # CHART
            # =========================
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

            # =========================
            # TABLE
            # =========================
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

            # =========================
            # REPORT DOWNLOAD
            # =========================
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

# =========================
# HISTORY
# =========================
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

# =========================
# FOOTER
# =========================
st.markdown("<br><br>", unsafe_allow_html=True)

st.caption("""
⚠️ Educational Project Disclaimer:
This AI system is for educational and research purposes only.
It does not replace professional medical diagnosis or treatment.
""")
