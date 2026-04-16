import streamlit as st
import re
import os
import smtplib
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from email.message import EmailMessage
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from streamlit.errors import StreamlitSecretNotFoundError

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        """
        Fallback no-op when python-dotenv is not installed.
        Streamlit secrets and existing environment variables can still be used.
        """
        return False

load_dotenv()

# --- Configuration & Styling ---
ST_PAGE_TITLE = "AI Resume Screening Tool"
ST_PAGE_ICON = "🎯"

def get_streamlit_secret(key, default=None):
    """
    Safely read a Streamlit secret without crashing when secrets.toml is missing.
    """
    try:
        return st.secrets.get(key, default)
    except StreamlitSecretNotFoundError:
        return default

def get_webhook_url():
    """
    Resolve webhook URL from environment or Streamlit secrets at runtime.
    """
    return (os.getenv("WEBHOOK_URL") or get_streamlit_secret("WEBHOOK_URL") or "").strip()

@st.cache_resource
def load_model():
    """
    Load the local sentence-transformer model.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def apply_custom_style():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Card-like containers - using theme variables */
    div.stBox, div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
        background-color: var(--secondary-background-color);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(151, 166, 195, 0.2);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Header styling */
    .stHeading h1 {
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    .stHeading h2 {
        font-weight: 700;
        color: var(--primary-color);
    }
    
    /* Buttons - using theme colors where possible */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    /* Custom Card for Top Candidates */
    .top-candidate-card {
        padding: 20px;
        border-radius: 12px;
        background: var(--secondary-background-color);
        border-left: 5px solid var(--primary-color);
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .rank-1 { border-left-color: #00DC82; }
    .rank-2 { border-left-color: #007BFF; }
    .rank-3 { border-left-color: #FFA500; }
    
    /* Progress bar styling */
    .stProgress [data-baseweb="progress-bar"] {
        height: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Basic common English stopwords
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
    'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'
}

# --- Backend Functions ---

def clean_text(text):
    """
    Lowercase, remove special characters, and remove stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    cleaned_words = [w for w in words if w not in STOPWORDS]
    return " ".join(cleaned_words)

def extract_text_from_pdf(file_obj):
    """
    Extract text from a PDF file-like object using pdfminer.six.
    """
    try:
        file_obj.seek(0)
        text = extract_text(BytesIO(file_obj.read()))
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_email(text):
    """
    Extract the first email address found in resume text.
    """
    if not text:
        return None

    match = re.search(r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', text)
    return match.group(1) if match else None

def extract_phone(text):
    """
    Extract the first plausible phone number from resume text.
    """
    if not text:
        return None

    match = re.search(r'(?:(?:\+91|91|0)[-\s]?)?(?:\(?\d{3,5}\)?[-\s]?)?\d{3,5}[-\s]?\d{3,5}[-\s]?\d{3,5}', text)
    if not match:
        return None

    phone = re.sub(r'\D', '', match.group(0))
    if len(phone) < 10:
        return None

    return phone

def extract_candidate_name(text, resume_file_name=""):
    """
    Infer a candidate name from resume text, with a filename fallback.
    """
    if text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:12]:
            normalized_line = re.sub(r"\s+", " ", line).strip(" ,.-:_")

            if not normalized_line:
                continue

            # Skip likely contact/details lines rather than a name header.
            if "@" in normalized_line or re.search(r"\d", normalized_line):
                continue

            words = normalized_line.split()
            if not 2 <= len(words) <= 4:
                continue

            if any(len(word) <= 1 for word in words):
                continue

            if any(not re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", word) for word in words):
                continue

            return " ".join(word.capitalize() for word in words)

    fallback_name = resume_file_name.rsplit(".", 1)[0] if resume_file_name else "Candidate"
    fallback_name = re.sub(r"[_\-]+", " ", fallback_name)
    fallback_name = re.sub(r"\b(resume|cv|profile|final|updated)\b", "", fallback_name, flags=re.IGNORECASE)
    fallback_name = re.sub(r"\s+", " ", fallback_name).strip(" ,.-:_")

    if fallback_name:
        return " ".join(word.capitalize() for word in fallback_name.split())

    return "Candidate"

def send_email(to_email, subject, message_body):
    """
    Send an HTML email using Gmail SMTP credentials from environment variables.
    """
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")

    if not email_user or not email_pass:
        return False, "Missing EMAIL_USER or EMAIL_PASS environment variables."

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = email_user
        msg["To"] = to_email
        msg.set_content("This email contains HTML content. Please view it in an HTML-compatible email client.")
        msg.add_alternative(message_body, subtype="html")

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(msg)

        return True, "Email sent successfully."
    except Exception as e:
        return False, str(e)

def log_to_google_sheet(name, email, phone, score, resume_name, status):
    """
    Log invitation activity to a Google Sheet webhook without interrupting the app.
    """
    webhook_url = get_webhook_url()

    if not webhook_url:
        return False, "Webhook URL not configured."

    payload = {
        "name": name,
        "candidate_name": name,
        "email": email,
        "phone": phone,
        "candidate_phone": phone,
        "score": score,
        "candidate_score": score,
        "resume": resume_name,
        "resume_name": resume_name,
        "resume_file_name": resume_name,
        "status": status,
    }

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.ok:
            return True, "Data logged to Google Sheets."

        response = requests.post(webhook_url, data=payload, timeout=10)
        if response.ok:
            return True, "Data logged to Google Sheets."

        return False, f"Logging failed with status {response.status_code}."
    except Exception as e:
        return False, f"Logging failed: {e}"

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title=ST_PAGE_TITLE, page_icon=ST_PAGE_ICON, layout="wide")
    apply_custom_style()
    
    # Sidebar
    with st.sidebar:
        st.title("🛡️Maple AI Resume Tool")
        st.info("Using AI Engine:\n`all-MiniLM-L6-v2`")
        st.markdown("---")
        st.write("This tool uses semantic analysis to compare resumes against your job description.")
        st.markdown("---")
        st.caption("v2.0 - Maple AI Resume Score Checker")

    # Header
    st.title(f"{ST_PAGE_ICON} AI Resume Screening Tool")
    st.markdown("##### Match candidates using AI-powered semantic analysis with high precision.")
    st.divider()

    # Initialize Session State for results
    if 'results_data' not in st.session_state:
        st.session_state.results_data = None
    if 'bulk_send_confirmed' not in st.session_state:
        st.session_state.bulk_send_confirmed = False
    if 'logging_status_message' not in st.session_state:
        st.session_state.logging_status_message = None
    if 'logging_status_type' not in st.session_state:
        st.session_state.logging_status_type = "info"

    # Layout: JD and Resume Uploads
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📄 Job Description")
        with st.container():
            jd_text = st.text_area("Paste the job requirements here...", height=250, placeholder="Requirements, skills, experience level...")
            jd_file = st.file_uploader("Or upload JD PDF", type=['pdf'], key="jd_upload")
            if jd_file:
                with st.spinner("Analyzing JD PDF..."):
                    extracted_jd = extract_text_from_pdf(jd_file)
                    if extracted_jd:
                        jd_text = extracted_jd
                        st.success("JD text successfully extracted!")

    with col2:
        st.markdown("### 📂 Upload Resumes")
        with st.container():
            resumes = st.file_uploader("Drop up to 50 resumes here", type=['pdf'], accept_multiple_files=True, help="Limit: 50 PDF files")
            if resumes:
                st.write(f"📊 **Total Resumes:** {len(resumes)}")
                if len(resumes) > 50:
                    st.error("⚠️ Maximum 50 resumes allowed.")
                    return
                # Show file list as tags (simplified)
                file_names = [f.name for f in resumes[:5]]
                st.caption(f"Files: {', '.join(file_names)}" + ("..." if len(resumes) > 5 else ""))

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Match and Reset Buttons
    m_col1, m_col2, m_col3 = st.columns([1, 1, 1])
    with m_col1:
        match_button = st.button("🚀 Match Resumes", type="primary", use_container_width=True)
    with m_col2:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.results_data = None
            st.rerun()

    # Matching Logic
    if match_button:
        if not jd_text.strip():
            st.error("Please provide a Job Description.")
        elif not resumes:
            st.error("Please upload at least one resume.")
        else:
            with st.spinner("🌟 AI is analyzing resumes..."):
                model = load_model()
                jd_cleaned = clean_text(jd_text)
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Building semantic model for JD...")
                jd_embedding = model.encode(jd_cleaned)
                
                results = []
                for i, resume_file in enumerate(resumes):
                    status_text.text(f"Analyzing {resume_file.name}...")
                    
                    resume_raw = extract_text_from_pdf(resume_file)
                    resume_cleaned = clean_text(resume_raw)
                    resume_embedding = model.encode(resume_cleaned)
                    extracted_email = extract_email(resume_raw)
                    extracted_phone = extract_phone(resume_raw)
                    candidate_name = extract_candidate_name(resume_raw, resume_file.name)
                    
                    if resume_embedding is not None:
                        similarity = cosine_similarity(
                            jd_embedding.reshape(1, -1),
                            resume_embedding.reshape(1, -1)
                        )[0][0]
                        
                        score = float(similarity) * 100
                        results.append({
                            "Resume Name": resume_file.name,
                            "Candidate Name": candidate_name,
                            "Score (%)": round(score, 2),
                            "Email": extracted_email,
                            "Phone": extracted_phone,
                            "Resume Text": resume_raw
                        })
                    
                    progress_bar.progress((i + 1) / len(resumes))
                
                status_text.text("✅ Analysis complete!")
                
                if results:
                    # Sort and Rank
                    sorted_results = sorted(results, key=lambda x: x["Score (%)"], reverse=True)
                    for rank, res in enumerate(sorted_results, 1):
                        res["Rank"] = rank
                    
                    # Save to session state
                    st.session_state.results_data = sorted_results
                    st.toast(f"Matching complete for {len(resumes)} resumes!", icon="✅")
                    st.rerun() # Refresh to show results properly
                else:
                    st.warning("No results were generated. Check your PDF contents.")

    # Persistent Results Rendering
    if st.session_state.results_data:
        results_df = st.session_state.results_data
        st.divider()
        st.header("📊 Screening Results")
        
        # Highlight Top 3
        st.markdown("#### 🏆 Top Candidates")
        t_cols = st.columns(3)
        ranks = ["🥇 Rank 1", "🥈 Rank 2", "🥉 Rank 3"]
        rank_classes = ["rank-1", "rank-2", "rank-3"]
        
        for i in range(min(3, len(results_df))):
            candidate = results_df[i]
            with t_cols[i]:
                st.markdown(f"""
                <div class="top-candidate-card {rank_classes[i]}">
                    <h4 style="margin:0; opacity: 0.8;">{ranks[i]}</h4>
                    <p style="font-weight:600; margin:10px 0;">{candidate['Resume Name']}</p>
                    <h2 style="margin:0; color: var(--primary-color);">{candidate['Score (%)']}%</h2>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Full Results Table with Filter
        st.markdown("#### 🔎 All Candidate Match Scores")
        
        score_filter = st.slider("Filter results by Score (%)", 0, 100, 50)
        
        final_df = pd.DataFrame(results_df)
        filtered_df = final_df[final_df["Score (%)"] >= score_filter]
        
        # Display Table
        display_cols = ["Rank", "Resume Name", "Score (%)"]
        st.dataframe(
            filtered_df[display_cols].style.format({"Score (%)": "{:.2f}%"})
            .background_gradient(subset=["Score (%)"], cmap="Greens"),
            use_container_width=True,
            hide_index=True
        )
        
        # Export Button
        csv = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name='resume_match_results.csv',
            mime='text/csv',
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.header("📧 Send Assessment Invitations")

        st.markdown("#### 📊 Logging Status")
        if st.session_state.logging_status_message:
            if st.session_state.logging_status_type == "success":
                st.success(st.session_state.logging_status_message)
            else:
                st.info(st.session_state.logging_status_message)

        invitation_threshold = st.slider("Minimum Score Threshold", 0, 100, 60)
        assessment_link = st.text_input("Assessment Link")
        email_subject = st.text_input("Subject", value="Assessment Invitation")
        email_template = st.text_area(
            "Message",
            value=(
                "<!DOCTYPE html>\n"
                "<html lang=\"en\">\n"
                "<head>\n"
                "  <meta charset=\"UTF-8\">\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                "  <title>Interview Invitation</title>\n"
                "</head>\n"
                "<body style=\"margin:0; padding:0; font-family: Arial, sans-serif;\">\n\n"
                "  <table width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" role=\"presentation\">\n"
                "    <tr>\n"
                "      <td align=\"center\">\n\n"
                "        <table width=\"600\" cellpadding=\"0\" cellspacing=\"0\" role=\"presentation\">\n\n"
                "          <tr>\n"
                "            <td>\n"
                "              <img src=\"https://maple.maplelearningsolutions.com/wp-content/uploads/2026/04/Purple-Gradient-Graphic-Designer-Email-Header.png\" alt=\"Header Image\" width=\"100%\" style=\"display:block; border:0; outline:none; text-decoration:none;\">\n"
                "            </td>\n"
                "          </tr>\n\n"
                "          <tr>\n"
                "            <td style=\"padding:20px; line-height:1.6;\">\n\n"
                "              <h2>Hi {name},</h2>\n\n"
                "              <p>Congratulations! We're pleased to inform you that, based on your resume and profile, you have been shortlisted for the next stage of our hiring process.</p>\n\n"
                "              <p>As a next step, we invite you to complete an assessment that will help us better understand your skills and suitability for the role. This assessment is an important part of our evaluation process.</p>\n\n"
                "              <p><strong>Assessment Details:</strong></p>\n\n"
                "              <ul>\n"
                "                <li><strong>Access Link:</strong> <a href=\"{assessment_link}\" style=\"text-decoration:none;\">Assessment Details</a></li>\n"
                "              </ul>\n\n"
                "              <p>We recommend completing the assessment in one sitting in a quiet environment with a stable internet connection. Kindly ensure that you submit your responses within the given timeframe.</p>\n\n"
                "              <p>If you have any questions or face any technical difficulties, feel free to reach out to us - we'll be happy to assist you.</p>\n\n"
                "              <p>We wish you the very best and look forward to reviewing your submission.</p>\n\n"
                "              <p>Best regards,<br>\n"
                "              Hiring Team</p>\n\n"
                "            </td>\n"
                "          </tr>\n\n"
                "          <tr>\n"
                "            <td style=\"background-color:#000000; padding:20px; color:#ffffff;\">\n\n"
                "              <table width=\"100%\" role=\"presentation\">\n"
                "                <tr>\n"
                "                  <td align=\"center\">\n"
                "                    <img src=\"https://maple.maplelearningsolutions.com/wp-content/uploads/2026/04/Logo_Maple-1.png\" alt=\"Company Logo\" style=\"display:block; border:0; width:120px; outline:none; text-decoration:none; margin-bottom:10px;\">\n"
                "                  </td>\n"
                "                </tr>\n"
                "              </table>\n\n"
                "              <table width=\"100%\" role=\"presentation\">\n"
                "                <tr>\n"
                "                  <td align=\"center\" style=\"padding:10px 0;\">\n"
                "                    <a href=\"https://www.maplelearningsolutions.com/\" style=\"margin:0 10px; text-decoration:none; color:#ffffff; border:none;\">Home</a>\n"
                "                    <a href=\"https://www.maplelearningsolutions.com/career\" style=\"margin:0 10px; text-decoration:none; color:#ffffff; border:none;\">Careers</a>\n"
                "                    <a href=\"mailto:info@maplelearningsolutions.com\" style=\"margin:0 10px; text-decoration:none; color:#ffffff; border:none;\">Contact</a>\n"
                "                  </td>\n"
                "                </tr>\n"
                "              </table>\n\n"
                "              <table width=\"100%\" role=\"presentation\">\n"
                "                <tr>\n"
                "                  <td align=\"center\" style=\"padding-top:10px;\">\n\n"
                "                    <a href=\"https://www.facebook.com/people/Maple-Learning-Solutions/61578896339989/\" style=\"text-decoration:none; border:none; display:inline-block; padding:0 6px;\">\n"
                "                      <img src=\"https://cdn-icons-png.flaticon.com/24/733/733547.png\" alt=\"Facebook\" style=\"display:block; border:0; outline:none;\">\n"
                "                    </a>\n\n"
                "                    <a href=\"https://www.youtube.com/channel/UCObkCM6XEdSA96dUwUbnV-g\" style=\"text-decoration:none; border:none; display:inline-block; padding:0 6px;\">\n"
                "                      <img src=\"https://cdn-icons-png.flaticon.com/24/1384/1384060.png\" alt=\"YouTube\" style=\"display:block; border:0; outline:none;\">\n"
                "                    </a>\n\n"
                "                    <a href=\"https://www.linkedin.com/company/maple-learning-solutions/posts/?feedView=all\" style=\"text-decoration:none; border:none; display:inline-block; padding:0 6px;\">\n"
                "                      <img src=\"https://cdn-icons-png.flaticon.com/24/174/174857.png\" alt=\"LinkedIn\" style=\"display:block; border:0; outline:none;\">\n"
                "                    </a>\n\n"
                "                    <a href=\"https://www.instagram.com/maple_learning_solutions\" style=\"text-decoration:none; border:none; display:inline-block; padding:0 6px;\">\n"
                "                      <img src=\"https://cdn-icons-png.flaticon.com/24/733/733558.png\" alt=\"Instagram\" style=\"display:block; border:0; outline:none;\">\n"
                "                    </a>\n\n"
                "                  </td>\n"
                "                </tr>\n"
                "              </table>\n\n"
                "            </td>\n"
                "          </tr>\n\n"
                "        </table>\n\n"
                "      </td>\n"
                "    </tr>\n"
                "  </table>\n\n"
                "</body>\n"
                "</html>"
            ),
            height=520
        )

        invitation_df = pd.DataFrame(results_df)
        shortlisted_df = invitation_df[invitation_df["Score (%)"] >= invitation_threshold].copy()
        valid_shortlisted_df = shortlisted_df[shortlisted_df["Email"].notna() & (shortlisted_df["Email"] != "")]
        invalid_shortlisted_df = shortlisted_df[shortlisted_df["Email"].isna() | (shortlisted_df["Email"] == "")]
        bulk_send_disabled = valid_shortlisted_df.empty

        st.caption(
            f"{len(valid_shortlisted_df)} eligible candidate(s) with valid email. "
            f"{len(invalid_shortlisted_df)} shortlisted candidate(s) missing email."
        )

        if not invalid_shortlisted_df.empty:
            missing_email_names = ", ".join(invalid_shortlisted_df["Resume Name"].tolist())
            st.warning(f"⚠️ Email not found for: {missing_email_names}")

        bulk_col1, bulk_col2 = st.columns([1, 2])
        with bulk_col1:
            if st.button(
                "Send to All Shortlisted Candidates",
                disabled=bulk_send_disabled,
                use_container_width=True,
                key="bulk_send_button"
            ):
                st.session_state.bulk_send_confirmed = True

        with bulk_col2:
            if st.session_state.bulk_send_confirmed:
                st.info("Bulk send is limited to 20 emails per click. Click the confirm button to continue.")
                if st.button(
                    "Confirm Bulk Send",
                    disabled=bulk_send_disabled,
                    use_container_width=True,
                    key="confirm_bulk_send_button"
                ):
                    candidates_to_send = valid_shortlisted_df.head(20).to_dict("records")
                    sent_count = 0
                    logged_count = 0

                    with st.spinner("Sending assessment invitations..."):
                        for candidate in candidates_to_send:
                            candidate_name = candidate.get("Candidate Name") or extract_candidate_name(
                                candidate.get("Resume Text", ""),
                                candidate["Resume Name"]
                            )
                            personalized_message = email_template.format(
                                name=candidate_name,
                                assessment_link=assessment_link
                            )
                            success, feedback = send_email(
                                candidate["Email"],
                                email_subject,
                                personalized_message
                            )

                            if success:
                                sent_count += 1
                                st.success(f"✅ Email sent successfully to {candidate['Email']}")
                                logged, log_feedback = log_to_google_sheet(
                                    candidate_name,
                                    candidate["Email"],
                                    candidate.get("Phone") or "",
                                    candidate["Score (%)"],
                                    candidate["Resume Name"],
                                    "Sent"
                                )
                            else:
                                st.error(f"❌ Failed to send to {candidate['Email']}: {feedback}")
                                logged, log_feedback = log_to_google_sheet(
                                    candidate_name,
                                    candidate["Email"],
                                    candidate.get("Phone") or "",
                                    candidate["Score (%)"],
                                    candidate["Resume Name"],
                                    "Failed"
                                )

                            if logged:
                                logged_count += 1

                        for candidate in invalid_shortlisted_df.head(20).to_dict("records"):
                            candidate_name = candidate.get("Candidate Name") or extract_candidate_name(
                                candidate.get("Resume Text", ""),
                                candidate["Resume Name"]
                            )
                            logged, log_feedback = log_to_google_sheet(
                                candidate_name,
                                candidate.get("Email") or "",
                                candidate.get("Phone") or "",
                                candidate["Score (%)"],
                                candidate["Resume Name"],
                                "No Email"
                            )
                            if logged:
                                logged_count += 1

                    st.info(f"{sent_count} emails sent successfully")
                    if logged_count:
                        st.session_state.logging_status_message = "Data logged to Google Sheets."
                        st.session_state.logging_status_type = "success"
                    else:
                        st.session_state.logging_status_message = log_feedback
                        st.session_state.logging_status_type = "info"
                    st.session_state.bulk_send_confirmed = False

        st.markdown("#### Candidate Actions")

        for candidate in invitation_df.to_dict("records"):
            is_eligible = candidate["Score (%)"] >= invitation_threshold
            has_email = bool(candidate.get("Email"))

            row_col1, row_col2, row_col3, row_col4 = st.columns([3, 1, 3, 2])
            with row_col1:
                st.write(candidate["Resume Name"])
            with row_col2:
                st.write(f"{candidate['Score (%)']:.2f}%")
            with row_col3:
                st.write(candidate["Email"] if has_email else "Email not found")
            with row_col4:
                send_disabled = not (is_eligible and has_email)
                if st.button(
                    "Send Invitation",
                    key=f"send_invite_{candidate['Rank']}",
                    disabled=send_disabled,
                    use_container_width=True
                ):
                    candidate_name = candidate.get("Candidate Name") or extract_candidate_name(
                        candidate.get("Resume Text", ""),
                        candidate["Resume Name"]
                    )
                    personalized_message = email_template.format(
                        name=candidate_name,
                        assessment_link=assessment_link
                    )

                    with st.spinner(f"Sending invitation to {candidate['Resume Name']}..."):
                        success, feedback = send_email(
                            candidate["Email"],
                            email_subject,
                            personalized_message
                        )

                    if success:
                        st.success(f"✅ Email sent successfully to {candidate['Email']}")
                        logged, log_feedback = log_to_google_sheet(
                            candidate_name,
                            candidate["Email"],
                            candidate.get("Phone") or "",
                            candidate["Score (%)"],
                            candidate["Resume Name"],
                            "Sent"
                        )
                    else:
                        st.error(f"❌ Failed to send to {candidate['Email']}: {feedback}")
                        logged, log_feedback = log_to_google_sheet(
                            candidate_name,
                            candidate["Email"],
                            candidate.get("Phone") or "",
                            candidate["Score (%)"],
                            candidate["Resume Name"],
                            "Failed"
                        )

                    if logged:
                        st.session_state.logging_status_message = "Data logged to Google Sheets."
                        st.session_state.logging_status_type = "success"
                    else:
                        st.session_state.logging_status_message = log_feedback
                        st.session_state.logging_status_type = "info"

            if is_eligible and not has_email:
                st.warning(f"⚠️ Email not found for {candidate['Resume Name']}")

if __name__ == "__main__":
    main()
