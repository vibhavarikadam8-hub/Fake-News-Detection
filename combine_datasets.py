import pandas as pd
import re

# --------------------------------
# 1. LOAD WELFAKE DATASET
# --------------------------------

wel = pd.read_csv(
    "dataset/WELFake_Dataset.csv",
    encoding="latin1"
)

print("WELFake loaded:", wel.shape)

# Keep title, text and label
wel = wel[["title", "text", "label"]].copy()

# Handle missing values
wel["title"] = wel["title"].fillna("")
wel["text"] = wel["text"].fillna("")

# Combine title and article
wel["news"] = (
    wel["title"].astype(str) + " " +
    wel["text"].astype(str)
)

wel = wel[["news", "label"]].copy()

# Original WELFake:
# 0 = FAKE, 1 = REAL

wel["label"] = pd.to_numeric(
    wel["label"], errors="coerce"
)

# Convert to our project convention:
# 0 = REAL, 1 = FAKE

wel["label"] = 1 - wel["label"]
wel["source_dataset"] = "WELFake"


# --------------------------------
# 2. LOAD IFND DATASET
# --------------------------------

ifnd = pd.read_csv(
    "dataset/IFND_cleaned.csv",
    encoding="latin1"
)

print("IFND loaded:", ifnd.shape)

# IFND already has standardized labels:
# 0 = REAL, 1 = FAKE

indian = ifnd[["news", "label"]].copy()

indian["source_dataset"] = "IFND"

# CHECK DATASET SOURCE COUNTS

print("\nWELFake source counts:")
print(wel["source_dataset"].value_counts())

print("\nIFND source counts:")
print(indian["source_dataset"].value_counts())



# --------------------------------
# 3. COMBINE DATASETS
# --------------------------------

combined = pd.concat(
    [wel, indian],
    ignore_index=True
)

# Remove missing and unexpected labels
combined["label"] = pd.to_numeric(
    combined["label"], errors="coerce"
)

combined = combined.dropna(
    subset=["news", "label"]
)

combined = combined[
    combined["label"].isin([0, 1])
].copy()

combined["label"] = combined["label"].astype(int)


# --------------------------------
# 4. CLEAN TEXT
# --------------------------------

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


combined["news"] = combined["news"].apply(clean_text)

combined = combined[
    combined["news"] != ""
].copy()


# --------------------------------
# 5. CHECK CONFLICTING DUPLICATES
# --------------------------------

combined["normalized_news"] = combined["news"]

label_conflicts = (
    combined.groupby("normalized_news")["label"]
    .nunique()
)

conflicting_news = label_conflicts[
    label_conflicts > 1
].index

print(
    "\nConflicting duplicate records:",
    len(conflicting_news)
)

# Exclude records whose identical text has conflicting labels
combined = combined[
    ~combined["normalized_news"].isin(conflicting_news)
].copy()


# --------------------------------
# 6. REMOVE DUPLICATES
# --------------------------------

before = len(combined)

combined = combined.drop_duplicates(
    subset=["normalized_news"],
    keep="first"
)

after = len(combined)

print("\nDuplicates removed:", before - after)


# --------------------------------
# 7. SAVE FINAL DATASET
# --------------------------------

combined = combined[
    ["news", "label", "source_dataset"]
]

combined.to_csv(
    "dataset/combined_news.csv",
    index=False
)


# --------------------------------
# 8. DISPLAY RESULTS
# --------------------------------

print("\n================================")
print("COMBINED DATASET RESULTS")
print("================================")

print("\nFinal Dataset Shape:")
print(combined.shape)

print("\nLabel Distribution:")
print(combined["label"].value_counts())

print("\nDataset Distribution:")
print(combined["source_dataset"].value_counts())

print("\nSample Records:")
print(combined.head(5).to_string())

print("\nCombined dataset saved successfully!")