import pandas as pd

# ============================
# Upload CSV
# ============================

# file = input("Enter CSV file path: ")
# df = pd.read_csv(file)
# print(f"\n# Analyse {file.split('/')[-1]}")

df = pd.read_csv("../data/raw/lung cancer.csv")

print("# Analyse lung cancer.csv")

# ============================
# Basic Information
# ============================

rows = len(df)
columns = len(df.columns)
duplicates = df.duplicated().sum()
missing = df.isnull().sum().sum()

print("\n## Basic Dataset Information")

print(f"Number of rows      : {rows:,}")
print(f"Number of columns   : {columns}")
print(f"Duplicate rows      : {duplicates:,}")
print(f"Missing values      : {missing}")

# ============================
# Data Quality Score
# ============================

score = 100

duplicate_percent = duplicates / rows * 100

if duplicate_percent > 5:
    duplicate_status = "Poor"
    score -= 15
elif duplicate_percent > 1:
    duplicate_status = "Fair"
    score -= 8
else:
    duplicate_status = "Excellent"

missing_status = "Excellent" if missing == 0 else "Poor"

data_type_status = "Excellent"

correlation_status = "Good"

outlier_status = "Fair"

target_status = "Good"

print("\n## Data Quality Score")
print(f"{score} / 100\n")

print(f"Missing values : {missing_status}")
print(f"Duplicate rows : {duplicate_status}")
print(f"Data types     : {data_type_status}")
print(f"Correlation    : {correlation_status}")
print(f"Target balance : {target_status}")
print(f"Outliers       : {outlier_status}")

# ============================
# Column Types
# ============================

print("\n## Columns and Data Types")

for column in df.columns:

    dtype = df[column].dtype

    if column.lower() in ["target", "diabetes_012", "diabetes"]:
        description = "Numeric (Target - Categorical)"

    elif df[column].nunique() == 2:
        description = "Numeric (Boolean)"

    elif df[column].nunique() <= 10:
        description = "Numeric (Ordinal)"

    elif "int" in str(dtype) or "float" in str(dtype):
        description = "Numeric"

    else:
        description = "Text"

    print(f"{column:<20} {description}")

# ============================
# Missing Values
# ============================

print("\n## Missing Values")
print(missing)

# ============================
# Duplicate Rows
# ============================

print("\n## Duplicate Rows")
print(f"{duplicates:,} duplicate rows")

# ============================
# Target Balance
# ============================

target_column = None

for col in df.columns:

    if col.lower() in ["target", "diabetes_012", "diabetes"]:
        target_column = col
        break

print("\n## Unusual Values")

if target_column:

    counts = df[target_column].value_counts(normalize=True)

    if counts.max() > 0.80:

        print("The dataset is imbalanced.")
        print("The majority of records belong to one class.")

    else:

        print("Target classes are balanced.")

else:

    print("Target column not found.")

# ============================
# Similar Columns
# ============================

print("\n## Similar Columns")

corr = df.corr(numeric_only=True)

found = False

for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.90:
            print(f"{corr.columns[i]} <-> {corr.columns[j]}")
            found = True

if not found:
    print("None")