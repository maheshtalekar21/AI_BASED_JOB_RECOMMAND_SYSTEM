# AI Job Recommendation System

## Project Overview
The AI Job Recommendation System is a machine-learning-based web application that recommends relevant job opportunities based on candidate skills and resume information.

The system uses Natural Language Processing (NLP), TF-IDF vectorization, and Cosine Similarity to compare candidate profiles with job descriptions.

## Key Features
- Resume upload using PDF.
- Automatic skill extraction.
- Personalized job recommendations.
- TF-IDF and Cosine Similarity matching.
- Matched and missing skill identification.
- Job ranking based on multiple preferences.
- Preferred job role, location, and job type filters.
- Interactive Streamlit dashboard.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- PyPDF

## Project Structure
```text
AI-Job-Recommendation/
│
├── dataset/
│   └── jobs_cleaned.csv
│
├── models/
│   ├── __init__.py
│   ├── job_recommender.py
│   ├── test_recommender.py
│   └── evaluate_recommender.py
│
├── utils/
│   ├── __init__.py
│   └── skill_extractor.py
│
├── app.py
├── preprocess.py
├── evaluation_data.py
├── evaluation_visualization.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project folder:

```bash
cd AI-Job-Recommendation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

## Recommendation Method

The system combines:
1. TF-IDF text similarity.
2. Candidate and job skill matching.
3. Preferred job role matching.
4. Preferred location matching.
5. Preferred job type matching.

## Future Enhancements
- Advanced NLP-based skill extraction.
- Improved recommendation ranking.
- Larger job datasets.
- Personalized career guidance.
- Cloud deployment.

## Project Purpose
This project demonstrates the practical application of Python, Machine Learning, NLP, and web development in solving a real-world job-matching problem.
