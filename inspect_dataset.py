
import pandas as pd
import os

# Dataset file path
file_path = "jobdata.csv"

print("================================")
print("   DATASET INSPECTION")
print("================================")

# Check whether file exists
if os.path.exists(file_path):

    print("\nDataset found successfully!")

    # Read only first 5 rows
    df = pd.read_csv(file_path, nrows=5)

    # Display column names
    print("\n1. COLUMN NAMES:")
    print(df.columns.tolist())

    # Display sample records
    print("\n2. FIRST 5 RECORDS:")
    print(df.head())

    # Display dataset information
    print("\n3. DATA TYPES:")
    print(df.dtypes)

    # Check missing values in a sample
    print("\n4. MISSING VALUES IN SAMPLE:")
    print(df.isnull().sum())

    # Display file size
    size = os.path.getsize(file_path) / (1024 * 1024)

    print("\n5. FILE SIZE:")
    print(round(size, 2), "MB")

else:
    print("Dataset file not found!")
    print("Check the file name and location.")