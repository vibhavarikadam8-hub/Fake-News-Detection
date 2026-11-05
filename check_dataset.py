import pandas as pd

file_path = "dataset/WELFake_Dataset.csv"

df = pd.read_csv(file_path)

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nLABEL DISTRIBUTION:")
print(df["label"].value_counts())

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

print("\nSAMPLE ARTICLES:")
print(df[["title", "text", "label"]].head(5).to_string())