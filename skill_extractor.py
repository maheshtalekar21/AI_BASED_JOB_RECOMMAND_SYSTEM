
import re

# Technical skills database
SKILLS = [
    "Python",
    "Java",
    "C++",
    "C#",
    "JavaScript",
    "HTML",
    "CSS",
    "React",
    "Angular",
    "Node.js",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "Excel",
    "Power BI",
    "Tableau",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "NLP",
    "Data Science",
    "Statistics",
    "Data Analysis",
    "Data Visualization",
    "Flask",
    "Django",
    "Streamlit",
    "Git",
    "GitHub",
    "AWS",
    "Azure",
    "Docker",
    "Linux",
    "Apache Spark"
]


def extract_skills(text):

    if not text:
        return []

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(skill.lower())
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, text):

            found_skills.append(skill)

    return sorted(set(found_skills))


# Test the function
if __name__ == "__main__":

    sample_resume = """
    I know Python, SQL, Machine Learning,
    Pandas, NumPy and Power BI.
    """

    result = extract_skills(sample_resume)

    print("Extracted Skills:")

    for skill in result:
        print("-", skill)

    print("\nTotal Skills:", len(result))
