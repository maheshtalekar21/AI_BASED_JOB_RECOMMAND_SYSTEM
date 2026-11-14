
import streamlit as st
import pandas as pd

from io import BytesIO
from pypdf import PdfReader
st.sidebar.subheader("Personalize Your Job Search")

preferred_roles = st.sidebar.multiselect(
    "Preferred Job Roles",
    [
        "Data Scientist",
        "Machine Learning Engineer",
        "Data Analyst",
        "Python Developer",
        "AI Engineer",
        "Software Engineer"
    ]
)

preferred_location = st.sidebar.text_input(
    "Preferred Location",
    placeholder="Example: Pune"
)

preferred_job_type = st.sidebar.selectbox(
    "Preferred Job Type",
    [
        "Any",
        "Full-time",
        "Part-time",
        "Internship",
        "Remote"
    ]
)

preferences = {
    "job_titles": preferred_roles,
    "location": preferred_location,
    "job_type": (
        "" if preferred_job_type == "Any"
        else preferred_job_type
    )
}
from models.job_recommender import (
    load_job_data,
    recommend_jobs
)

from utils.skill_extractor import extract_skills


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="💼",
    layout="wide"
)


# ==========================================
# 2. CUSTOM DASHBOARD DESIGN
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
}

.dashboard-title {
    font-size: 32px;
    font-weight: bold;
    color: #183153;
}

.dashboard-subtitle {
    color: #64748b;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD ML MODEL
# ==========================================

@st.cache_resource
def get_model():

    jobs, vectorizer, job_vectors = load_job_data()

    return jobs, vectorizer, job_vectors


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def get_column_value(row, candidates, default="Not specified"):

    for column in candidates:

        if column in row.index:

            value = row[column]

            if pd.notna(value) and str(value).strip():
                return str(value)

    return default


def extract_resume_text(uploaded_file):

    pdf = PdfReader(
        BytesIO(uploaded_file.getvalue())
    )

    text = ""

    for page in pdf.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    return text


# ==========================================
# 5. DASHBOARD HEADER
# ==========================================

st.markdown(
    '<div class="dashboard-title">'
    '💼 AI Job Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Smart job matching using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 6. SIDEBAR
# ==========================================

st.sidebar.title("Dashboard Menu")

st.sidebar.info(
    "Upload your resume to discover "
    "relevant job opportunities."
)

st.sidebar.markdown("### Recommendation Settings")

top_n = st.sidebar.selectbox(
    "Number of recommendations",
    [5, 10, 15],
    index=0
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Powered by TF-IDF and Cosine Similarity"
)


# ==========================================
# 7. LOAD DATASET
# ==========================================

try:

    with st.spinner("Loading job dataset and ML model..."):

        jobs, vectorizer, job_vectors = get_model()

    st.success("Job dataset and recommendation model loaded!")

except Exception as error:

    st.error(f"Model loading failed: {error}")

    st.stop()


# ==========================================
# 8. DATASET OVERVIEW
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Job Records",
        f"{len(jobs):,}"
    )

with col2:

    st.metric(
        "TF-IDF Features",
        f"{len(vectorizer.get_feature_names_out()):,}"
    )

with col3:

    st.metric(
        "Recommendation Algorithm",
        "TF-IDF + Cosine"
    )


st.divider()


# ==========================================
# 9. RESUME UPLOAD
# ==========================================

st.header("📄 Upload Your Resume")

col1, col2 = st.columns([2, 1])

with col1:

    candidate_name = st.text_input(
        "Candidate Name",
        placeholder="Enter your name"
    )

    uploaded_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )

with col2:

    st.info(
        """
        **Supported format:** PDF

        Upload a text-based PDF resume.

        Your uploaded PDF is processed in memory
        and is not saved to disk.
        """
    )


# ==========================================
# 10. ANALYZE RESUME
# ==========================================

if st.button(
    "🚀 Analyze Resume & Find Jobs",
    type="primary",
    use_container_width=True
):

    if not candidate_name.strip():

        st.warning("Please enter your name.")

    elif uploaded_file is None:

        st.warning("Please upload your PDF resume.")

    else:

        try:

            with st.spinner("Analyzing your resume..."):

                resume_text = extract_resume_text(
                    uploaded_file
                )

                if not resume_text.strip():

                    st.error(
                        "No readable text found. "
                        "Please upload a text-based PDF."
                    )

                    st.stop()

                # Extract skills
                candidate_skills = extract_skills(
                    resume_text
                )

                if not candidate_skills:

                    st.warning(
                        "No matching skills detected. "
                        "Please check your resume."
                    )

                    st.stop()

                # Generate recommendations
                recommendations = recommend_jobs(
                    candidate_skills=candidate_skills,
                    jobs=jobs,
                    vectorizer=vectorizer,
                    job_vectors=job_vectors,
                    top_n=top_n,
                    preferences=preferences
                )

            # Store results for dashboard display
            st.session_state["candidate_name"] = candidate_name
            st.session_state["candidate_skills"] = candidate_skills
            st.session_state["recommendations"] = recommendations

            st.success("Resume analyzed successfully!")

        except Exception as error:

            st.error(f"Resume analysis failed: {error}")


# ==========================================
# 11. DISPLAY ANALYSIS RESULTS
# ==========================================

if "recommendations" in st.session_state:

    recommendations = st.session_state["recommendations"]

    candidate_skills = st.session_state["candidate_skills"]

    st.divider()

    st.header(
        f"Welcome, {st.session_state['candidate_name']}!"
    )

    # Candidate skills
    st.subheader("🧠 Extracted Technical Skills")

    st.write(", ".join(candidate_skills))

    st.metric(
        "Total Skills Detected",
        len(candidate_skills)
    )

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview",
        "💼 Job Recommendations",
        "🎯 Skill Gap Analysis"
    ])


    # ======================================
    # TAB 1: OVERVIEW
    # ======================================

    with tab1:

        st.subheader("Job Matching Overview")

        st.write(
            "Your resume has been compared with "
            "the available job descriptions."
        )

        chart_data = recommendations[
            ["title", "match_score"]
        ].copy()

        chart_data = chart_data.rename(
            columns={
                "title": "Job Title",
                "match_score": "Similarity Score"
            }
        )

        st.bar_chart(
            chart_data.set_index("Job Title")
        )

        st.metric(
            "Jobs Recommended",
            len(recommendations)
        )

        st.caption(
            "Similarity scores measure textual similarity, "
            "not hiring probability."
        )


    # ======================================
    # TAB 2: JOB RECOMMENDATIONS
    # ======================================

    with tab2:

        st.subheader("Recommended Job Opportunities")

        search_keyword = st.text_input(
            "Search job title",
            placeholder="Example: Data Scientist"
        )

        location_candidates = [
            "job_location",
            "location",
            "city",
            "work_location"
        ]

        location_column = next(
            (
                col for col in recommendations.columns
                if col.lower().strip() in location_candidates
            ),
            None
        )

        filtered = recommendations.copy()

        if search_keyword:

            filtered = filtered[
                filtered["title"].str.contains(
                    search_keyword,
                    case=False,
                    na=False
                )
            ]

        if location_column:

            locations = sorted(
                recommendations[location_column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_location = st.selectbox(
                "Filter by location",
                ["All Locations"] + locations
            )

            if selected_location != "All Locations":

                filtered = filtered[
                    filtered[location_column].astype(str)
                    == selected_location
                ]

        if filtered.empty:

            st.warning(
                "No jobs match your selected filters. "
                "Try another search."
            )

        else:

            for rank, (_, job) in enumerate(
                filtered.iterrows(),
                start=1
            ):

                with st.container(border=True):

                    st.subheader(
                        f"{rank}. {job['title']}"
                    )

                    st.metric(
                        "Text Similarity Score",
                        f"{job['match_score']}%"
                    )

                    st.write(
                        "**Location:**",
                        get_column_value(
                            job,
                            [
                                "job_location",
                                "location",
                                "city",
                                "work_location"
                            ]
                        )
                    )

                    st.write(
                        "**Job Type:**",
                        get_column_value(
                            job,
                            ["work_type", "job_type"]
                        )
                    )

                    st.write(
                        "**Experience Level:**",
                        get_column_value(
                            job,
                            ["level", "experience_level"]
                        )
                    )

                    if job["missing_skills"]:

                        st.write(
                            "**Potential Missing Skills:**",
                            ", ".join(job["missing_skills"])
                        )

                    else:

                        st.success(
                            "No missing skills detected "
                            "from the current skills database."
                        )

            # Download recommendation results
            csv_data = filtered.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Recommendations CSV",
                data=csv_data,
                file_name="job_recommendations.csv",
                mime="text/csv"
            )


    # ======================================
    # TAB 3: SKILL GAP ANALYSIS
    # ======================================

    with tab3:

        st.subheader("Candidate Skill Gap Analysis")

        candidate_set = {
            skill.lower()
            for skill in candidate_skills
        }

        required_skills = set()

        for skills in recommendations["detected_job_skills"]:

            required_skills.update(skills)

        missing_skills = sorted([
            skill for skill in required_skills
            if skill.lower() not in candidate_set
        ])

        st.write("### Your Current Skills")

        st.write(", ".join(candidate_skills))

        st.write("### Potential Skills to Develop")

        if missing_skills:

            for skill in missing_skills:

                st.write(f"📌 {skill}")

        else:

            st.success(
                "No additional skills detected "
                "in the current recommendations."
            )

        st.metric(
            "Potential Skill Gaps",
            len(missing_skills)
        )

        st.info(
            "Skill gaps are based on detected words "
            "in the recommended job descriptions. "
            "They are suggestions, not confirmed employer requirements."
        )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Job Recommendation System | "
    "Developed using Python, Pandas, Scikit-learn and Streamlit"
)
