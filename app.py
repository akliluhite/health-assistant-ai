import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

# Set premium clinical flat minimalist page configurations
st.set_page_config(
    page_title="AI Clinical Diagnostic Portal",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# FIXED IMAGE SOURCE: High-reliability public URL designed for live web application background layering
BACKGROUND_IMAGE_URL = "https://unsplash.com"

# Premium Material UI Clinical CSS System with full-screen structural image layering
st.markdown(f"""
    <style>
    /* Full-Screen Structural Background Image configurations */
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.88)), url("{BACKGROUND_IMAGE_URL}");
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
        border: 1px solid #e2e8f0 !important;
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
    }}
    .stButton>button[data-testid="baseButton-primary"]:hover {{
        background-color: #115e59 !important;
    }}
    
    /* Clean White Metric Component Blocks */
    .clinical-metric {{
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #e2e8f0;
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
        border: 1px solid #e2e8f0;
        padding: 16px;
        border-radius: 8px;
        margin-top: 12px;
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    </style>
""", unsafe_allow_html=True)

# CSS-based Corporate Medical Header Box (Clean background layer configuration)
st.markdown("""
    <div style="background-color: rgba(15, 118, 110, 0.08); padding: 25px; border-radius: 8px; text-align: left; margin-bottom: 25px; border: 1px solid rgba(15, 118, 110, 0.15);">
        <h2 style="color: #0f766e; margin: 0; font-family: 'Inter', sans-serif; font-size: 1.5rem; font-weight: 700;">🏥 Clinical Informatics Intelligence Center</h2>
        <p style="color: #115e59; margin: 6px 0 0 0; font-size: 0.9rem; font-weight: 500; letter-spacing: 0.5px;">Machine Learning Statistical Inference Gateway</p>
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
    "Jaundice": {"desc": "A yellowing of the skin and eyes caused by high levels of bilirubin.", "specialist": "Hepatologist / Gastroenterologist", "treatment": "Prioritize rest, eliminate completely all alcoholic pathways, monitor liver functions."},
    "Malaria": {"desc": "A disease caused by a plasmodium parasite, transmitted by the bite of infected mosquitoes.", "specialist": "Infectious Disease Specialist", "treatment": "Seek immediate antimalarial pharmacotherapy, monitor hydration levels carefully."},
    "Chicken pox": {"desc": "A highly contagious viral infection causing an itchy, blister-like rash on the skin.", "specialist": "General Physician / Pediatrician", "treatment": "Rest, maintain clean skin, use anti-itch lotions, monitor internal body temperature flags."},
    "Dengue": {"desc": "A mosquito-borne viral disease occurring in tropical and subtropical areas.", "specialist": "Infectious Disease Specialist", "treatment": "Maximize clean hydration intake profiles, take paracetamol for tracking pain, avoid ibuprofen options."},
    "Typhoid": {"desc": "A bacterial infection spread through contaminated food and water.", "specialist": "Infectious Disease Specialist", "treatment": "Complete complete antibiotic prescription metrics, consume fully pasteurized foods."},
    "Common Cold": {"desc": "A common viral infection of the nose and throat.", "specialist": "General Physician", "treatment": "Maintain vocal rest, ingest warm fluid configurations, track respiratory markers."},
    "Pneumonia": {"desc": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid.", "specialist": "Pulmonologist / General Physician", "treatment": "Follow structural antibiotic schedules, rest continuously, monitor blood oxygen metrics."},
    "Heart attack": {"desc": "A medical emergency where blood flow to a part of the heart is blocked.", "specialist": "Cardiologist", "treatment": "Call global emergency response units instantly, chew emergency aspirin protocols."},
    "Acne": {"desc": "A skin condition that occurs when hair follicles become plugged with oil and dead skin cells.", "specialist": "Dermatologist", "treatment": "Clean facial features gently using non-comedogenic elements, apply specialized skincare lines."},
    "Urinary tract infection": {"desc": "An infection in any part of the urinary system, including kidneys, ureters, bladder, and urethra.", "specialist": "Urologist / General Physician", "treatment": "Increase internal water consumption limits, acquire target prescription treatments."},
    "Psoriasis": {"desc": "A condition in which skin cells build up and form scales and itchy, dry patches.", "specialist": "Dermatologist", "treatment": "Apply thick emollient ointments, avoid extreme environment humidity shifts."},
    "Impetigo": {"desc": "A highly contagious skin infection that causes sores, mainly around the nose and mouth.", "specialist": "Dermatologist / General Physician", "treatment": "Apply topical prescription medication arrays, sanitize clothing items separately."}
}

HOSPITAL_DIRECTORY = {
    "Dermatologist": {"dept": "Dermatology & Skin Sciences Clinic", "hotline": "+1 (555) 019-2831", "floor": "Building B, 3rd Floor"},
    "Allergist / Immunologist": {"dept": "Allergy Research & Immunology Institute", "hotline": "+1 (555) 014-9922", "floor": "Main West Wing, 2nd Floor"},
    "Gastroenterologist": {"dept": "Gastrointestinal Health & Endoscopy Lab", "hotline": "+1 (555) 017-8811", "floor": "Outpatient Pavilion, Ground Floor"},
    "Hepatologist / Gastroenterologist": {"dept": "Advanced Liver & Hepatobiliary Care Center", "hotline": "+1 (555) 012-3344", "floor": "Medical Tower A, 4th Floor"},
    "General Physician / Allergist": {"dept": "Family Medicine & Urgent Screening Facility", "hotline": "+1 (555) 011-5500", "floor": "Emergency Annex, Suite 10"},
    "Infectious Disease Specialist": {"dept": "Specialized Pathogen & Infectious Management Wing", "hotline": "+1 (555) 016-7788", "floor": "Isolation Pavilion, Restricted Zone C"},
    "Endocrinologist": {"dept": "Metabolic Disorders & Endocrinology Center", "hotline": "+1 (555) 015-4433", "floor": "Building C, 1st Floor"},
    "Pulmonologist / Allergist": {"dept": "Respiratory & Asthma Critical Care Center", "hotline": "+1 (555) 018-2211", "floor": "Main Building, 5th Floor"},
    "Cardiologist": {"dept": "Cardiovascular Sciences & Critical Intervention Center", "hotline": "+1 (555) 019-9900", "floor": "Cardiac Wing, Ground Floor Location"},
    "Neurologist": {"dept": "Neurological Assessment & Stroke Evaluation Lab", "hotline": "+1 (555) 013-1122", "floor": "Neuro-Sciences Block, 3rd Floor"},
    "Orthopedic Surgeon / Neurologist": {"dept": "Spinal Health & Orthopedic Rehabilitation Unit", "hotline": "+1 (555) 014-5544", "floor": "West Wing, Ground Floor Annex"},
    "Neurologist / Neurosurgeon": {"dept": "Neuro-Trauma & Comprehensive Stroke Center", "hotline": "+1 (555) 012-8877", "floor": "Neuro Intensive Care Unit, 2nd Floor"},
    "General Physician / Pediatrician": {"dept": "Family & Pediatric Care Clinic", "hotline": "+1 (555) 017-3311", "floor": "Building B, 1st Floor"},
    "Pulmonologist": {"dept": "Advanced Thoracic Disease & Pulmonology Unit", "hotline": "+1 (555) 016-1155", "floor": "Main Building, 4th Floor Wing"},
    "General Physician": {"dept": "Primary Health Assessment Portal", "hotline": "+1 (555) 011-2233", "floor": "Main Entry Lounge, Room 101"},
    "Pulmonologist / General Physician": {"dept": "Acute Respiratory Evaluation Unit", "hotline": "+1 (555) 018-4455", "floor": "Main Tower, 5th Floor East"},
    "General Surgeon / Proctologist": {"dept": "Colorectal Health & General Surgical Suites", "hotline": "+1 (555) 013-6677", "floor": "Surgical Pavilion, 2nd Floor"},
    "Vascular Surgeon": {"dept": "Vascular Anomalies & Venous Circulatory Clinic", "hotline": "+1 (555) 015-9988", "floor": "Building A, Suite 4B"},
    "Rheumatologist / Orthopedist": {"dept": "Joint Inflammation & Rheumatology Center", "hotline": "+1 (555) 014-2200", "floor": "Outpatient Pavilion, 2nd Floor"},
    "Rheumatologist": {"dept": "Autoimmune & Clinical Rheumatology Center", "hotline": "+1 (555) 014-7766", "floor": "Building C, 2nd Floor Suite"},
    "ENT Specialist / Neurologist": {"dept": "Vestibular Disorders & Otolaryngology Unit", "hotline": "+1 (555) 016-8899", "floor": "Main Building, 2nd Floor North"},
    "Urologist / General Physician": {"dept": "Renal Health & Comprehensive Urology Clinic", "hotline": "+1 (555) 012-4411", "floor": "Medical Tower B, Ground Floor Annex"},
    "Dermatologist / General Physician": {"dept": "Acute Skin Lesion & Dermatology Unit", "hotline": "+1 (555) 019-3322", "floor": "Building B, 3rd Floor East"}
}

# Session History Logging Array Initialization
if "history_log" not in st.session_state:
    st.session_state.history_log = []

# Patient Registry Demographic Sidebar Panel Configuration
st.sidebar.header("Patient Entry Profiles")
patient_age = st.sidebar.slider("Patient Age:", min_value=1, max_value=100, value=30)
patient_gender = st.sidebar.selectbox("Biological Sex:", ["Male", "Female", "Other"])

st.sidebar.markdown("---")
st.sidebar.subheader("Case Duration Matrix")
symptom_duration = st.sidebar.selectbox("Symptoms Persisting For:", ["Less than 24 Hours", "1 to 3 Days", "More than a Week"])

# Workspace Title Headers
st.title("AI Clinical Diagnosis Portal")
st.markdown("<div class='sub-heading'>Isolate specific patient symptom indicators to compute machine learning probability classification paths.</div>", unsafe_allow_html=True)

if model_ready:
    clean_features = [f.replace("_", " ").title() for f in features]
    
    if "symptom_key" not in st.session_state:
        st.session_state.symptom_key = 0

    selected_clean = st.multiselect(
        "Identify Patient Manifestations:", 
        clean_features, 
        placeholder="Filter available indicator variables...",
        key=f"symptoms_{st.session_state.symptom_key}"
    )
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        submit_btn = st.button("Run Diagnostic Analysis", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Clear Input Node", use_container_width=True):
            st.session_state.symptom_key += 1
            st.rerun()

    if submit_btn:
        if not selected_clean:
            st.warning("Please isolate manifestation variables inside the selector tool before continuing.")
        else:
            start_latency_time = time.perf_counter()
            
            input_matrix = np.zeros((1, len(features)))
            for clean_sym in selected_clean:
                raw_name = clean_sym.lower().replace(" ", "_")
                if raw_name in features:
                    idx = features.index(raw_name)
                    input_matrix[0, idx] = 1
            
            prediction_raw = model.predict(input_matrix)
            prediction = str(prediction_raw).strip()
            
            probabilities = model.predict_proba(input_matrix).flatten()
            classes = [str(c).strip() for c in model.classes_]
            
            if prediction in classes:
                class_idx = classes.index(prediction)
                confidence = probabilities[class_idx] * 100
            else:
                confidence = 0.0
                
            end_latency_time = time.perf_counter()
            processing_latency = (end_latency_time - start_latency_time) * 1000

            bytes_symptom_count = len(selected_clean)
            if bytes_symptom_count <= 2 and "Week" not in symptom_duration:
                risk_tier = "Mild Case Load"
                risk_color = "#0f766e"
            elif bytes_symptom_count <= 5 and "Week" not in symptom_duration:
                risk_tier = "Moderate Case Load"
                risk_color = "#d97706"
            else:
                risk_tier = "Urgent Intervention Needed"
                risk_color = "#dc2626"

            # Log record into memory array
            st.session_state.history_log.append({
                "Condition": prediction,
                "Certainty": f"{confidence:.1f}%",
                "Risk Level": risk_tier,
                "Symptoms Count": bytes_symptom_count
            })

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.markdown(f"""
                    <div class="clinical-metric">
                        <div class="clinical-label">Primary Diagnostic Target</div>
                        <div class="clinical-value" style="color: #0f766e;">{prediction}</div>
                    </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                    <div class="clinical-metric">
                        <div class="clinical-label">Model Certainty Score</div>
                        <div class="clinical-value" style="color: #475569;">{confidence:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                    <div class="clinical-metric">
                        <div class="clinical-label">Triage Priority</div>
                        <div class="clinical-value" style="color: {risk_color}; font-size:1.1rem;">{risk_tier}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"<p style='color: #475569; font-size: 0.8rem; margin-top: 10px; text-align: right; font-weight: 600;'>Inference Latency: {processing_latency:.3f} ms</p>", unsafe_allow_html=True)
            
            desc_text = "Conditional indicator profiles pending engine knowledge base extension."
            spec_text = "General Physician"
            treatment_advice = "Follow baseline supportive home care directives, monitor condition progression metrics, rest appropriately."
            
            if prediction in DISEASE_INFO:
                desc_text = DISEASE_INFO[prediction]["desc"]
                spec_text = DISEASE_INFO[prediction]["specialist"]
                if "treatment" in DISEASE_INFO[prediction]:
                    treatment_advice = DISEASE_INFO[prediction]["treatment"]
                
            st.markdown(f"""
                <div class="clinical-info-bin">
                    <strong>Medical Pathogen Overview:</strong> {desc_text}
                </div>
                <div class="clinical-info-bin">
                    📍 <strong>Recommended Specialist Directive:</strong> Referral directed towards a <strong>{spec_text}</strong>.
                </div>
                <div class="clinical-info-bin">
                    🛡️ <strong>Initial General Management Measures:</strong> {treatment_advice}
                </div>
            """, unsafe_allow_html=True)

            # Facility Department Referral Finder Output
            st.markdown("<br><h5>🏢 Facility Care Unit Referral Finder</h5>", unsafe_allow_html=True)
            if spec_text in HOSPITAL_DIRECTORY:
                dir_info = HOSPITAL_DIRECTORY[spec_text]
                st.markdown(f"""
                    <div class="clinical-info-bin" style="background-color: #ffffff; margin-top: 5px;">
                        🏢 <strong>Target Department:</strong> {dir_info['dept']}<br>
                        📍 <strong>Internal Clinic Wing:</strong> {dir_info['floor']}<br>
                        📞 <strong>Direct Unit Contact Line:</strong> <span style="color: #0f766e; font-weight: bold;">{dir_info['hotline']}</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Hospital routing path details are pending centralized updates.")

            if patient_age < 12 and "Reaction" in prediction:
                st.error("🚨 **Pediatric Warning Indicator:** Selected profile markers display hyper-sensitivity indicators to conventional chemical therapies. Avoid self-treatment.")
            elif patient_age > 65:
                st.warning("⚠️ **Geriatric Metric Warning:** Clearance rates for primary drug pathways are slowed in patients over 65. Clinical review is advised.")

            st.markdown("<br><h5>Secondary Statistical Variant Distributions</h5>", unsafe_allow_html=True)
            top_indices = np.argsort(probabilities)[::-1][:3]
            chart_data = pd.DataFrame({
                "Condition Classes": [classes[i] for i in top_indices],
                "Confidence Match Score (%)": [probabilities[i] * 100 for i in top_indices]
            })
            st.bar_chart(chart_data, x="Condition Classes", y="Confidence Match Score (%)", color="#0f766e")

            secondary_indices = np.argsort(probabilities)[::-1][1:6]
            matrix_df = pd.DataFrame({
                "Alternative Diagnostic Targets": [classes[i] for i in secondary_indices],
                "Statistical Probability Match": [f"{probabilities[i] * 100:.2f}%" for i in secondary_indices]
            })
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)

            report_content = f"""# CLINICAL DATA SCIENCE INFERENCE LOG REPORT
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

    # Session Diagnostic History Audit Log component
    if st.session_state.history_log:
        st.markdown("<br><h5>📜 System Session Diagnostic Audit Logs</h5>", unsafe_allow_html=True)
        history_df = pd.DataFrame(st.session_state.history_log[::-1])
        st.dataframe(history_df, use_container_width=True, hide_index=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("⚠️ **Educational Project Disclaimer:** This system functions strictly as a data-science exercise using training datasets. It does not replace professional medical evaluations, clinical triage plans, or medical advice.")
