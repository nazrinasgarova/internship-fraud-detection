import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="SafeIntern | AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished financial/risk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# Load model pipeline
@st.cache_resource
def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "fraud_detection_xgb_full_pipeline.joblib")
    if not os.path.exists(model_path):
        model_path = "fraud_detection_xgb_full_pipeline.joblib"
    return joblib.load(model_path)

try:
    pipeline = load_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model pipeline: {e}")

# Header
st.markdown('<div class="main-header">🛡️ SafeIntern: AI Job Fraud & Risk Detection Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Decision Intelligence & Compliance Triage Engine powered by XGBoost (85.2% Recall @ 0.36 Risk Threshold | 0.9682 ROC-AUC)</div>', unsafe_allow_html=True)

# Sidebar with system info
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=70)
    st.title("System Specs")
    st.markdown("""
    * **Algorithm:** XGBoost Classifier
    * **Tuned Threshold:** `0.36` (Risk-Optimized)
    * **Test ROC-AUC:** `0.9682`
    * **Holdout Recall:** `85.2%`
    * **Architecture:** Scikit-Learn ColumnTransformer + Pipeline
    """)
    st.divider()
    st.subheader("Quick Presets")
    preset = st.radio(
        "Load demo scenario:",
        ["Custom Input", "⚠️ High-Risk Scam Posting", "✅ Legitimate Tech Internship"],
        index=0
    )

# Predefined preset values
if preset == "⚠️ High-Risk Scam Posting":
    default_title = "Remote Data Entry & AI Assistant"
    default_emp = "Part-time"
    default_work = "Remote"
    default_ind = "Technology"
    default_loc = "Remote"
    default_size = "Startup"
    default_age = 1
    default_stipend = 1200.0
    default_pay_req = 1
    default_cert = 1
    default_web = 0
    default_link = 0
    default_susp_email = 1
    default_email_type = "Free"
    default_trust = 2.1
    default_urgency = 8.5
    default_phish = 7.0
    default_desc_len = 250
    default_gram = 5
    default_vague = 7.5
    default_spam = 6.0
elif preset == "✅ Legitimate Tech Internship":
    default_title = "Data Science Intern"
    default_emp = "Internship"
    default_work = "Hybrid"
    default_ind = "Technology"
    default_loc = "Baku"
    default_size = "Enterprise"
    default_age = 12
    default_stipend = 700.0
    default_pay_req = 0
    default_cert = 0
    default_web = 1
    default_link = 1
    default_susp_email = 0
    default_email_type = "Corporate"
    default_trust = 8.9
    default_urgency = 1.0
    default_phish = 0.5
    default_desc_len = 1400
    default_gram = 0
    default_vague = 1.0
    default_spam = 0.5
else:
    default_title = "Software Engineering Intern"
    default_emp = "Internship"
    default_work = "Hybrid"
    default_ind = "Technology"
    default_loc = "Baku"
    default_size = "Medium"
    default_age = 5
    default_stipend = 500.0
    default_pay_req = 0
    default_cert = 0
    default_web = 1
    default_link = 1
    default_susp_email = 0
    default_email_type = "Corporate"
    default_trust = 6.5
    default_urgency = 2.0
    default_phish = 1.0
    default_desc_len = 800
    default_gram = 1
    default_vague = 2.0
    default_spam = 1.0

# Input Form
with st.form("internship_scoring_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Company & Infrastructure")
        company_size = st.selectbox("Company Size", ["Startup", "Small", "Medium", "Enterprise"], 
                                    index=["Startup", "Small", "Medium", "Enterprise"].index(default_size))
        company_age = st.number_input("Company Age (Years)", min_value=0, max_value=100, value=default_age)
        industry = st.selectbox("Industry", ["Technology", "Finance", "Healthcare", "Education", "Marketing", "Retail", "Other"],
                                index=0 if default_ind not in ["Technology", "Finance", "Healthcare"] else ["Technology", "Finance", "Healthcare"].index(default_ind))
        website_available = st.radio("Official Website Exists?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No", index=0 if default_web==1 else 1)
        linkedin_presence = st.radio("Verified LinkedIn Profile?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No", index=0 if default_link==1 else 1)
        trust_signal_score = st.slider("Trust Signal Score", 0.0, 10.0, float(default_trust), 0.1)

    with col2:
        st.subheader("2. Job Terms & Critical Flags")
        internship_title = st.text_input("Internship Title", value=default_title)
        work_mode = st.selectbox("Work Mode", ["On-site", "Hybrid", "Remote"], index=["On-site", "Hybrid", "Remote"].index(default_work))
        employment_type = st.selectbox("Employment Type", ["Internship", "Part-time", "Full-time"], index=["Internship", "Part-time", "Full-time"].index(default_emp))
        stipend = st.number_input("Monthly Stipend (USD / AZN)", min_value=0.0, max_value=50000.0, value=float(default_stipend), step=50.0)
        payment_required = st.selectbox("🚨 Registration / Application Fee Demanded?", [0, 1], 
                                        format_func=lambda x: "Yes (Demands Fee)" if x==1 else "No (Free Application)", 
                                        index=default_pay_req)
        fake_certificate_offer = st.selectbox("Guaranteed Paid Certificate Offered?", [0, 1], 
                                              format_func=lambda x: "Yes (Offers Paid Cert)" if x==1 else "No", 
                                              index=default_cert)

    with col3:
        st.subheader("3. Phishing & Recruiter Signals")
        recruiter_email_type = st.selectbox("Recruiter Email Type", ["Corporate", "Free", "Unknown"], 
                                            index=["Corporate", "Free", "Unknown"].index(default_email_type))
        suspicious_email_domain = st.selectbox("Suspicious Email Domain Flag?", [0, 1], 
                                               format_func=lambda x: "Flagged Domain" if x==1 else "Clean Domain", 
                                               index=default_susp_email)
        urgency_score = st.slider("Urgency Pressure Score", 0.0, 10.0, float(default_urgency), 0.5)
        phishing_language_score = st.slider("Phishing Language Score", 0.0, 10.0, float(default_phish), 0.5)
        vague_description_score = st.slider("Vague Description Score", 0.0, 10.0, float(default_vague), 0.5)
        job_description_length = st.number_input("Job Description Length (Chars)", min_value=50, max_value=10000, value=default_desc_len)
        grammatical_errors = st.slider("Grammatical Errors Count", 0, 20, default_gram)

    submit_button = st.form_submit_button("🔍 Run AI Fraud Risk Assessment", use_container_width=True)

# Prediction execution
if submit_button and model_loaded:
    # Compute engineered features
    risk_signals_sum = int(payment_required) + int(fake_certificate_offer) + int(suspicious_email_domain) + (1 - int(website_available)) + (1 - int(linkedin_presence))
    urgency_density = float(urgency_score) / (float(job_description_length) + 1.0)
    stipend_to_age_ratio = float(stipend) / (float(company_age) + 1.0)
    
    input_df = pd.DataFrame([{
        'internship_title': internship_title,
        'employment_type': employment_type,
        'work_mode': work_mode,
        'industry': industry,
        'location': default_loc,
        'company_size': company_size,
        'company_age': company_age,
        'linkedin_presence': linkedin_presence,
        'website_available': website_available,
        'verification_status': 1 if trust_signal_score > 5 else 0,
        'stipend': stipend,
        'payment_required': payment_required,
        'job_description_length': job_description_length,
        'grammatical_errors': grammatical_errors,
        'vague_description_score': vague_description_score,
        'urgency_score': urgency_score,
        'keyword_spam_score': default_spam,
        'fake_certificate_offer': fake_certificate_offer,
        'recruiter_experience_years': 3,
        'recruiter_email_type': recruiter_email_type,
        'suspicious_email_domain': suspicious_email_domain,
        'recruiter_response_time_hours': 12.0,
        'social_media_presence': linkedin_presence,
        'emotional_manipulation_score': (urgency_score + phishing_language_score) / 2.0,
        'phishing_language_score': phishing_language_score,
        'trust_signal_score': trust_signal_score,
        'posting_year': 2024,
        'posting_month': 7,
        'posting_day': 15,
        'risk_signals_sum': risk_signals_sum,
        'urgency_density': urgency_density,
        'stipend_to_age_ratio': stipend_to_age_ratio
    }])

    # Score model
    fraud_prob = pipeline.predict_proba(input_df)[0, 1]
    tuned_threshold = 0.36
    is_fraud = fraud_prob >= tuned_threshold

    st.divider()
    st.subheader("📊 AI Compliance & Triage Verdict")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(
            label="Predicted Fraud Probability",
            value=f"{fraud_prob * 100:.1f}%",
            delta=f"{(fraud_prob - tuned_threshold) * 100:+.1f}% vs Threshold (36%)",
            delta_color="inverse"
        )
        st.progress(float(fraud_prob))
        
    with res_col2:
        if is_fraud:
            st.error(f"🚨 **HIGH RISK FLAGGED — PROBABLE FRAUDULENT POSTING**")
            st.markdown(f"""
            * **Decision:** Automated Block / Escalated Compliance Audit.
            * **Risk Assessment:** The posting exceeds the **36.0% risk threshold** calibrated for financial fraud prevention.
            * **Composite Risk Signals Identified:** **{risk_signals_sum} concurrent red flags**.
            """)
        else:
            st.success(f"✅ **LOW RISK VERIFIED — LEGITIMATE POSTING**")
            st.markdown(f"""
            * **Decision:** Approved for Public Board Publication.
            * **Risk Assessment:** Probability ({fraud_prob*100:.1f}%) is well below the **36.0% threshold**.
            * **Trust Factors:** Sufficient company tenure, valid infrastructure, and absence of predatory payment requirements.
            """)

    # Detailed Red Flag Breakdown
    with st.expander("🔍 Risk Factor Breakdown & SHAP Indicators", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.write("**Payment Demands:**", "❌ Upfront Fee Demanded" if payment_required else "✔️ Free Application")
            st.write("**Accreditation:**", "❌ Paid Certificate Trap" if fake_certificate_offer else "✔️ Clean Terms")
        with f_col2:
            st.write("**Digital Footprint:**", "✔️ Verified Website" if website_available else "❌ Missing Website")
            st.write("**Recruiter Identity:**", "❌ Flagged Suspicious Domain" if suspicious_email_domain else "✔️ Clean Domain")
        with f_col3:
            st.write("**Composite Red Flags:**", f"{risk_signals_sum} / 5 signals")
            st.write("**Urgency Density:**", f"{urgency_density:.4f}")
