import pandas as pd

# Load IFND dataset
df = pd.read_csv(
    "dataset/IFND.csv",
    encoding="latin1"
)

# Keep required columns
df = df[["Statement", "Label", "Category", "Web"]].copy()

# Rename columns
df.rename(columns={
    "Statement": "news",
    "Label": "label",
    "Category": "category",
    "Web": "source"
}, inplace=True)

# Remove missing news and labels
df.dropna(subset=["news", "label"], inplace=True)

# Clean text
df["news"] = df["news"].astype(str).str.strip()

# Standardize labels
df["label"] = df["label"].astype(str).str.strip().str.lower()

# Convert labels into numeric values
df["label"] = df["label"].map({
    "true": 0,
    "fake": 1
})

# Remove unexpected labels
df.dropna(subset=["label"], inplace=True)

# Remove empty news
df = df[df["news"] != ""]

# Remove duplicate news
df.drop_duplicates(subset=["news"], inplace=True)

# Convert labels to integers
df["label"] = df["label"].astype(int)

# Save cleaned dataset
df.to_csv("dataset/IFND_cleaned.csv", index=False)

print("IFND cleaning completed!")

print("\nFinal Shape:")
print(df.shape)

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nSample Records:")
print(df[["news", "label"]].head())