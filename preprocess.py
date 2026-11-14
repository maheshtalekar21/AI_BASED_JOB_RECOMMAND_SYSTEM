
import pandas as pd
import os
import re

print("=" * 50)
print("   JOB DATA CLEANING AND PREPROCESSING")
print("=" * 50)

# 1. Load original dataset
file_path = "jobdata.csv"

df = pd.read_csv(file_path, low_memory=False)

print("\nOriginal Dataset Shape:", df.shape)

# 2. Remove duplicate records
df.drop_duplicates(subset="job_id", inplace=True)

print("After Removing Duplicates:", df.shape)

# 3. Handle missing values
df["level"] = df["level"].fillna("Not specified")

df["job_domain"] = df["job_domain"].fillna("Not specified")

# 4. Clean text columns
def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces and newlines
    text = re.sub(r"\s+", " ", text)

    # Remove unwanted special characters
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)

    return text.strip()


# 5. Clean job descriptions
df["clean_description"] = df["description"].apply(clean_text)

# 6. Clean job titles
df["clean_title"] = df["title"].apply(clean_text)

# 7. Create combined text for ML
df["search_text"] = (
    df["clean_title"] + " " +
    df["clean_description"] + " " +
    df["job_domain"].apply(clean_text) + " " +
    df["level"].apply(clean_text)
)

# 8. Remove records with empty search text
df = df[df["search_text"].str.strip() != ""]

# 9. Select useful columns
selected_columns = [
    "job_id",
    "title",
    "description",
    "job_location",
    "work_type",
    "level",
    "job_domain",
    "compensation",
    "pay_period",
    "remote_allowed",
    "clean_description",
    "clean_title",
    "search_text"
]

cleaned_df = df[selected_columns].copy()

# 10. Save cleaned dataset
os.makedirs("dataset", exist_ok=True)

output_path = "dataset/jobs_cleaned.csv"

cleaned_df.to_csv(output_path, index=False)

# 11. Display final results
print("\nDATA CLEANING COMPLETED!")

print("\nFinal Dataset Shape:", cleaned_df.shape)

print("\nRemaining Missing Values:")
print(cleaned_df.isnull().sum())

print("\nFirst Five Cleaned Records:")
print(cleaned_df[["title", "job_domain", "level"]].head())

print("\nCleaned Dataset Saved At:", output_path)

print("\nAll preprocessing steps completed successfully!")
