import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

# Set premium clinical dark layout configuration metrics
st.set_page_config(
    page_title="AI Clinical Diagnostic Node",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Premium Slate Dark Mode Theme Layer Injection
st.markdown("""
    <style>
    /* Main Dark View Frame Base */
    .stApp {
        background-color: #0f172a !important;
        color: #f8fafc !important;
    }
    
    /* Clean Minimal Header Structure */
    h1 {
        color: #38bdf8 !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    .sub-heading {
        color: #94a3b8 !important;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
    }
    
    /* Sidebar styling overrides */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label {
        color: #f8fafc !important;
    }

    /* Core Input Panel Content Card Element Wrapper */
    div[data-testid="stVerticalBlock"] > div:has(div.stMultiSelect) {
        background: #1e293b !important;
        padding: 35px !important;
        border-radius: 16px !important;
        border: 1px solid #334155 !important;
        box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.3) !important;
    }

    /* Primary Interactive Trigger Button Formatting */
    .stButton>button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }
    
    /* Premium Styled Output Display Metric Blocks */
    .clinical-metric {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
        border-top: 4px solid #38bdf8;
    }
    .clinical-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.75px;
    }
    .clinical-value {
        font-size: 1.35rem;
        color: #ffffff;
        font-weight: 700;
        margin-top: 6px;
    }
    
    /* General Text Information Callout Bins */
    .clinical-info-bin {
        background-color: #111827;
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 10px;
        margin-top: 12px;
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.5;
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
    "Jaundice": {"desc": "A yellowing of the skin and eyes caused by high levels of bilirubin.", "specialist": "Hepatologist / Gastroenterologist", "treatment": "Prioritize rest, eliminate completely all alcoholic pathways, monitor liver functions."},
    "Malaria": {"desc": "A disease caused by a plasmodium parasite, transmitted by the bite of infected mosquitoes.", "specialist": "Infectious Disease Specialist", "treatment": "Seek immediate antimalarial pharmacotherapy, monitor hydration levels carefully."},
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

# Patient Registry Demographic Sidebar Panel Configuration
st.sidebar.header("📋 Patient Profile Registry")
patient_age = st.sidebar.slider("Patient Age:", min_value=1, max_value=100, value=30)
patient_gender = st.sidebar.selectbox("Biological Sex:", ["Male", "Female", "Other"])

# Feature 2: Severity Multi-Class Indicator Modification Dropdowns
st.sidebar.markdown("---")
st.sidebar.subheader("⏳ Case Timeline Severity")
symptom_duration = st.sidebar.selectbox("Symptoms Persisting For:", ["Less than 24 Hours", "1 to 3 Days", "More than a Week"])

st.title("AI Clinical Diagnosis Portal")
st.markdown("<div class='sub-heading'>Isolate observed physiological indicator configurations to evaluate classifier prediction vectors.</div>", unsafe_allow_html=True)

if model_ready:
    clean_features = [f.replace("_", " ").title() for f in features]
    
    if "symptom_key" not in st.session_state:
        st.session_state.symptom_key = 0

    selected_clean = st.multiselect(
        "Identify Patient Manifestations:", 
        clean_features, 
        placeholder="Search specific indicator rows...",
        key=f"symptoms_{st.session_state.symptom_key}"
    )
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        submit_btn = st.button("Execute Inference Matrix", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Clear Input Node", use_container_width=True):
            st.session_state.symptom_key += 1
            st.rerun()

    if submit_btn:
        if not selected_clean:
            st.warning("Please isolate structural manifestation vectors inside the multi-select workspace node first.")
        else:
            # Feature 4: Model Compute Latency Start Tracker
            start_latency_time = time.perf_counter()
            
            input_matrix = np.zeros((1, len(features)))
            for clean_sym in selected_clean:
                raw_name = clean_sym.lower().replace(" ", "_")
                if raw_name in features:
                    idx = features.index(raw_name)
                    input_matrix[0, idx] = 1
            
            # Formulate 2D target metrics
            prediction_raw = model.predict(input_matrix)
            prediction = str(prediction_raw[0]).strip()
            
            probabilities = model.predict_proba(input_matrix).flatten()
            classes = [str(c).strip() for c in model.classes_]
            
            if prediction in classes:
                class_idx = classes.index(prediction)
                confidence = probabilities[class_idx] * 100
            else:
                confidence = 0.0
                
            # Model Compute Latency Finish Evaluation 
            end_latency_time = time.perf_counter()
            processing_latency = (end_latency_time - start_latency_time) * 1000

            # Scale risk profiles dynamically against symptom load and user timeline context
            symptom_count = len(selected_clean)
            if symptom_count <= 2 and "Week" not in symptom_duration:
                risk_tier = "Mild"
                risk_color = "#10b981"
            elif symptom_count <= 5 and "Week" not in symptom_duration:
                risk_tier = "Moderate"
                risk_color = "#f59e0b"
            else:
                risk_tier = "Urgent / Elevated"
                risk_color = "#f43f5e"

            # Render Primary Visual Dash Blocks
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(f"""
                    <div class="clinical-metric">
                        <div class="clinical-label">Inferred Classification</div>
                        <div class="clinical-value" style="color: #38bdf8;">{prediction}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                    <div class="clinical-metric" style="border-top-color: #10b981;">
                        <div class="clinical-label">Model Certitude</div>
                        <div class="clinical-value" style="color: #10b981;">{confidence:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                    <div class="clinical-metric" style="border-top-color: {risk_color};">
                        <div class="clinical-label">Triage Priority Status</div>
                        <div class="clinical-value" style="color: {risk_color};">{risk_tier}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Display Compute Latency Metrics
            st.markdown(f"<p style='color: #64748b; font-size: 0.85rem; margin-top: 10px; text-align: right;'>⏱️ Pipeline processing latency: {processing_latency:.3f} ms</p>", unsafe_allow_html=True)
            
            # Fetch dictionary info values cleanly
            desc_text = "Conditional indicator profiles pending engine knowledge base extension."
            spec_text = "Consultation with a General Medicine department lead is recommended for initial validation routing."
            treatment_advice = "Follow baseline supportive home care directives, monitor condition progression metrics, rest appropriately."
            
            if prediction in DISEASE_INFO:
                desc_text = DISEASE_INFO[prediction]["desc"]
                spec_text = DISEASE_INFO[prediction]["specialist"]
                if "treatment" in DISEASE_INFO[prediction]:
                    treatment_advice = DISEASE_INFO[prediction]["treatment"]
                
            st.markdown(f"""
                <div class="clinical-info-bin">
                    <strong>Medical Condition Overview:</strong> {desc_text}
                </div>
                <div class="clinical-info-bin" style="border-left: 4px solid #f59e0b;">
                    📍 <strong>Assigned Clinical Routing Vector:</strong> Referral recommended to a <strong>{spec_text}</strong>.
                </div>
            """, unsafe_allow_html=True)

            # Feature 3: Actionable First-Aid Treatment Recommendation Block View
            st.markdown(f"""
                <div class="clinical-info-bin" style="border-left: 4px solid #10b981; background-color: #064e3b; color: #a7f3d0;">
                    🛡️ <strong>First-Line General Guidance Measures:</strong> {treatment_advice}
                </div>
            """, unsafe_allow_html=True)

            # Demographics Safety Rules Callouts
            if patient_age < 12 and "Reaction" in prediction:
                st.error("🚨 **Pediatric Warning Indicator:** Selected profile markers display hyper-sensitivity indicators to conventional chemical therapies. Avoid self-treatment.")
            elif patient_age > 65:
                st.warning("⚠️ **Geriatric Metric Warning:** Clearance rates for primary drug pathways are slowed in patients over 65. Clinical review is advised.")

            # Variant Distribution Graph
            st.markdown("<br><h5 style='color: #38bdf8;'>Statistical Secondary Variant Analysis</h5>", unsafe_allow_html=True)
            top_indices = np.argsort(probabilities)[::-1][:3]
            chart_data = pd.DataFrame({
                "Condition Vector Class": [classes[i] for i in top_indices],
                "Confidence Match Score (%)": [probabilities[i] * 100 for i in top_indices]
            })
            st.bar_chart(chart_data, x="Condition Vector Class", y="Confidence Match Score (%)", color="#0ea5e9")

            # Secondary Variant Data Table Frame
            secondary_indices = np.argsort(probabilities)[::-1][1:6]
            matrix_df = pd.DataFrame({
                "Alternative Target Conditions": [classes[i] for i in secondary_indices],
                "Statistical Probability Match": [f"{probabilities[i] * 100:.2f}%" for i in secondary_indices]
            })
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)

            # Dynamic Markdown Report Builder File Stream
            report_content = f"""# CLINICAL DATA SCIENCE INFerence LOG REPORT
- **Patient Profile Demographics:** {patient_age} Years Old ({patient_gender})
- **Reported Timeline Status:** {symptom_duration}
- **Assigned Triage Risk Tier Level:** {risk_tier}

### TRACKED INDICATION MANIFESTATIONS:
{', '.join(selected_clean)}

### CLASSIFIER DISCOVERY:
- **Inferred Primary Condition Label:** {prediction}
- **Algorithmic Confidence Matrix Score:** {confidence:.2f}%
- **Assigned Clinical Specialist Unit:** {spec_text}
- **First-Line Supportive Directives:** {treatment_advice}
"""
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Export Report Summary Data File",
                data=report_content,
                file_name="Clinical_Diagnostic_Report.md",
                mime="text/markdown",
                use_container_width=True
            )

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("⚠️ **Educational Project Disclaimer:** This system functions strictly as a data-science exercise using training datasets. It does not replace professional medical evaluations, clinical triage plans, or medical advice.")
