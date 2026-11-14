
from models.job_recommender import (
    load_job_data,
    recommend_jobs
)

from evaluation_data import evaluation_cases


# =========================================
# 1. EVALUATION CONFIGURATION
# =========================================

TOP_K = 5


# =========================================
# 2. CALCULATE EVALUATION METRICS
# =========================================

def calculate_metrics(
    recommended_titles,
    relevant_titles,
    total_relevant,
    k=5
):

    recommended = [
        str(title).lower().strip()
        for title in recommended_titles[:k]
    ]

    relevant = [
        str(title).lower().strip()
        for title in relevant_titles
    ]

    # Match relevant titles using manually
    # labelled title keywords.
    hits = sum(
        any(label in title for label in relevant)
        for title in recommended
    )

    precision = hits / k if k else 0

    recall = (
        hits / total_relevant
        if total_relevant > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    # Mean Reciprocal Rank for this query
    reciprocal_rank = 0

    for rank, title in enumerate(recommended, start=1):

        if any(label in title for label in relevant):

            reciprocal_rank = 1 / rank
            break

    return precision, recall, f1, reciprocal_rank, hits


# =========================================
# 3. LOAD RECOMMENDATION MODEL
# =========================================

print("=" * 55)
print("AI JOB RECOMMENDATION - MODEL EVALUATION")
print("=" * 55)

jobs, vectorizer, job_vectors = load_job_data()


# =========================================
# 4. EVALUATE EACH TEST CASE
# =========================================

all_precision = []
all_recall = []
all_f1 = []
all_mrr = []

for case_number, case in enumerate(
    evaluation_cases,
    start=1
):

    skills = case["candidate_skills"]

    relevant_titles = case["relevant_titles"]

    # Find all relevant dataset jobs
    # using manually defined title labels.
    relevant_mask = jobs["title"].str.lower().apply(
        lambda title: any(
            label.lower() in title
            for label in relevant_titles
        )
    )

    total_relevant = int(relevant_mask.sum())

    if total_relevant == 0:

        print(
            f"\nTest Case {case_number}: "
            "No labelled relevant titles found."
        )

        print(
            "Update relevant_titles in evaluation_data.py."
        )

        continue

    # Generate recommendations
    recommendations = recommend_jobs(
        skills,
        jobs,
        vectorizer,
        job_vectors,
        top_n=TOP_K
    )

    recommended_titles = recommendations[
        "title"
    ].tolist()

    precision, recall, f1, mrr, hits = calculate_metrics(
        recommended_titles,
        relevant_titles,
        total_relevant,
        TOP_K
    )

    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)
    all_mrr.append(mrr)

    print(f"\nTEST CASE {case_number}")

    print("Candidate Skills:", skills)

    print("Relevant Jobs Available:", total_relevant)

    print("\nRecommended Jobs:")

    for rank, (_, job) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        print(
            rank,
            job["title"],
            "- Similarity:",
            str(job["match_score"]) + "%"
        )

    print("\nEvaluation Results:")

    print(f"Precision@{TOP_K}: {precision:.3f}")
    print(f"Recall@{TOP_K}: {recall:.3f}")
    print(f"F1@{TOP_K}: {f1:.3f}")
    print(f"MRR: {mrr:.3f}")

    print("Relevant recommendations:", hits)


# =========================================
# 5. FINAL AVERAGE RESULTS
# =========================================

if all_precision:

    print("\n" + "=" * 55)
    print("FINAL MODEL PERFORMANCE")
    print("=" * 55)

    print(
        "Evaluated test cases:",
        len(all_precision)
    )

    print(
        f"Average Precision@{TOP_K}: "
        f"{sum(all_precision) / len(all_precision):.3f}"
    )

    print(
        f"Average Recall@{TOP_K}: "
        f"{sum(all_recall) / len(all_recall):.3f}"
    )

    print(
        f"Average F1@{TOP_K}: "
        f"{sum(all_f1) / len(all_f1):.3f}"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{sum(all_mrr) / len(all_mrr):.3f}"
    )

else:

    print(
        "\nNo test cases could be evaluated. "
        "Please check your relevance labels."
    )
