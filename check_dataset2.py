import pandas as pd

df = pd.read_csv(
    "dataset/IFND.csv",
    encoding="latin1"
)

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head().to_string())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nLabel Distribution:")
print(df.iloc[:, -1].value_counts())