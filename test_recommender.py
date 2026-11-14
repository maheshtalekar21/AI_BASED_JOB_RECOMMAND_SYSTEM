from models.job_recommender import (
    load_job_data,
    recommend_jobs
)


# Load dataset and model
jobs, vectorizer, job_vectors = load_job_data()

# Example candidate skills
candidate_skills = [
    "python",
    "machine learning",
    "sql",
    "pandas",
    "data analysis"
]

# Candidate preferences
preferences = {
    "job_titles": [
        "Data Scientist",
        "Machine Learning Engineer"
    ],
    "location": "Pune",
    "job_type": "Full-time"
}

# Generate recommendations
recommendations = recommend_jobs(
    candidate_skills=candidate_skills,
    jobs=jobs,
    vectorizer=vectorizer,
    job_vectors=job_vectors,
    top_n=10,
    preferences=preferences
)

# Display results
print("\n===== PERSONALIZED JOB RECOMMENDATIONS =====\n")

for _, job in recommendations.iterrows():

    print("Rank:", job["recommendation_rank"])
    print("Job Title:", job["title"])
    print("Match Score:", str(job["match_score"]) + "%")
    print("Matched Skills:", job["matched_skills"])
    print("Missing Skills:", job["missing_skills"])
    print("Reason:", job["recommendation_reason"])

    print("-" * 60)