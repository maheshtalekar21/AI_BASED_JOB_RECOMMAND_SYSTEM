import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.skill_extractor import extract_skills


# -------------------------------------------------
# 1. LOAD AND PREPARE JOB DATASET
# -------------------------------------------------

def load_job_data(file_path="dataset/jobs_cleaned.csv"):

    jobs = pd.read_csv(file_path)

    # Find job title column
    title_columns = [
        "title", "job_title", "job title",
        "position", "designation"
    ]

    title_col = next(
        (col for col in title_columns if col in jobs.columns),
        None
    )

    if title_col is None:
        raise ValueError(
            "Job title column not found. Check your dataset columns."
        )

    # Find job description / skills column
    text_columns = [
        "search_text", "description",
        "job_description", "job description",
        "skills", "requirements"
    ]

    text_col = next(
        (col for col in text_columns if col in jobs.columns),
        None
    )

    if text_col is None:
        raise ValueError(
            "Job description or skills column not found."
        )

    jobs["title"] = jobs[title_col].fillna("").astype(str)
    jobs["search_text"] = (
        jobs[text_col].fillna("").astype(str)
        + " "
        + jobs["title"]
    )

    # Extract skills required by each job
    jobs["detected_job_skills"] = jobs["search_text"].apply(
        lambda text: extract_skills(text) or []
    )

    # Create TF-IDF model
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
    )

    job_vectors = vectorizer.fit_transform(jobs["search_text"])

    return jobs, vectorizer, job_vectors


# -------------------------------------------------
# 2. TEXT NORMALIZATION
# -------------------------------------------------

def normalize_text(value):

    if value is None:
        return ""

    return str(value).strip().lower()


def find_column(dataframe, possible_columns):

    for col in possible_columns:
        if col in dataframe.columns:
            return col

    return None


# -------------------------------------------------
# 3. ADVANCED JOB RECOMMENDATION
# -------------------------------------------------

def recommend_jobs(
    candidate_skills,
    jobs,
    vectorizer,
    job_vectors,
    top_n=5,
    preferences=None
):

    if jobs.empty:
        return pd.DataFrame()

    preferences = preferences or {}

    # Accept skills as either a list or a string
    if isinstance(candidate_skills, str):
        candidate_skill_list = extract_skills(candidate_skills) or []
    else:
        candidate_skill_list = candidate_skills or []

    candidate_skill_list = list({
        normalize_text(skill)
        for skill in candidate_skill_list
        if normalize_text(skill)
    })

    if not candidate_skill_list:
        raise ValueError(
            "No skills detected. Please provide valid candidate skills."
        )

    # Convert candidate skills into text
    candidate_text = " ".join(candidate_skill_list)

    # Calculate TF-IDF similarity
    candidate_vector = vectorizer.transform([candidate_text])

    similarity_scores = cosine_similarity(
        candidate_vector,
        job_vectors
    ).flatten()

    # Identify optional dataset columns
    location_col = find_column(
        jobs,
        ["location", "job_location", "city"]
    )

    job_type_col = find_column(
        jobs,
        ["job_type", "employment_type", "type"]
    )

    # Read user preferences
    preferred_titles = preferences.get("job_titles", [])
    preferred_location = normalize_text(
        preferences.get("location", "")
    )
    preferred_job_type = normalize_text(
        preferences.get("job_type", "")
    )

    if isinstance(preferred_titles, str):
        preferred_titles = [preferred_titles]

    preferred_titles = [
        normalize_text(title) for title in preferred_titles
        if normalize_text(title)
    ]

    results = jobs.copy()

    # -------------------------------------------------
    # 4. CALCULATE SKILL MATCHING
    # -------------------------------------------------

    matched_skills_list = []
    missing_skills_list = []
    skill_scores = []

    for job_skills in results["detected_job_skills"]:

        job_skill_set = {
            normalize_text(skill)
            for skill in (job_skills or [])
            if normalize_text(skill)
        }

        candidate_skill_set = set(candidate_skill_list)

        matched = sorted(candidate_skill_set & job_skill_set)
        missing = sorted(job_skill_set - candidate_skill_set)

        if job_skill_set:
            skill_score = (
                len(matched) / len(job_skill_set)
            ) * 100
        else:
            skill_score = 0

        matched_skills_list.append(matched)
        missing_skills_list.append(missing)
        skill_scores.append(skill_score)

    results["matched_skills"] = matched_skills_list
    results["missing_skills"] = missing_skills_list
    results["skill_score"] = skill_scores

    # -------------------------------------------------
    # 5. PREFERENCE MATCHING
    # -------------------------------------------------

    title_scores = []
    location_scores = []
    job_type_scores = []

    for _, row in results.iterrows():

        # Preferred job title
        title = normalize_text(row["title"])

        title_score = 100 if any(
            preferred in title or title in preferred
            for preferred in preferred_titles
        ) else 0

        title_scores.append(title_score)

        # Preferred location
        location_score = 0

        if preferred_location and location_col:
            job_location = normalize_text(row[location_col])

            if preferred_location in job_location:
                location_score = 100

        location_scores.append(location_score)

        # Preferred job type
        job_type_score = 0

        if preferred_job_type and job_type_col:
            job_type = normalize_text(row[job_type_col])

            if preferred_job_type in job_type:
                job_type_score = 100

        job_type_scores.append(job_type_score)

    results["title_preference_score"] = title_scores
    results["location_preference_score"] = location_scores
    results["job_type_preference_score"] = job_type_scores

    # -------------------------------------------------
    # 6. FINAL PERSONALIZED SCORE
    # -------------------------------------------------

    results["similarity_score"] = similarity_scores * 100

    results["match_score"] = (
        results["similarity_score"] * 0.35
        + results["skill_score"] * 0.45
        + results["title_preference_score"] * 0.10
        + results["location_preference_score"] * 0.05
        + results["job_type_preference_score"] * 0.05
    ).round(2)

    # -------------------------------------------------
    # 7. GENERATE RECOMMENDATION EXPLANATION
    # -------------------------------------------------

    def explain_recommendation(row):

        reasons = []

        if row["matched_skills"]:
            reasons.append(
                "Matched skills: "
                + ", ".join(row["matched_skills"][:5])
            )

        if row["title_preference_score"] > 0:
            reasons.append("Matches your preferred job role")

        if row["location_preference_score"] > 0:
            reasons.append("Matches your preferred location")

        if row["job_type_preference_score"] > 0:
            reasons.append("Matches your preferred job type")

        if not reasons:
            reasons.append("Matches your resume content")

        return " | ".join(reasons)

    results["recommendation_reason"] = results.apply(
        explain_recommendation,
        axis=1
    )

    # Rank jobs by final score
    results = results.sort_values(
        by="match_score",
        ascending=False
    )

    results["recommendation_rank"] = range(1, len(results) + 1)

    return results.head(top_n).reset_index(drop=True)