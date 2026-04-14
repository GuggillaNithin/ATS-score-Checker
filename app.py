import streamlit as st
import re
import numpy as np
import pandas as pd
from io import BytesIO
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- Configuration & Styling ---
ST_PAGE_TITLE = "AI Resume Screening Tool"
ST_PAGE_ICON = "🎯"

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
                    
                    if resume_embedding is not None:
                        similarity = cosine_similarity(
                            jd_embedding.reshape(1, -1),
                            resume_embedding.reshape(1, -1)
                        )[0][0]
                        
                        score = float(similarity) * 100
                        results.append({
                            "Resume Name": resume_file.name,
                            "Score (%)": round(score, 2)
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

if __name__ == "__main__":
    main()
