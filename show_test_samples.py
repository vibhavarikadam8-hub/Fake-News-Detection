import pandas as pd
from sklearn.model_selection import train_test_split

# Load combined dataset
df = pd.read_csv("dataset/combined_news.csv")

test_data = []

# Recreate the same 80-20 split used during training
for source in df["source_dataset"].unique():

    source_data = df[df["source_dataset"] == source]

    train, test = train_test_split(
        source_data,
        test_size=0.20,
        random_state=42,
        stratify=source_data["label"]
    )

    test_data.append(test)

# Combine testing records
test_df = pd.concat(test_data)

# Display actual test examples
for label, name in [(0, "REAL NEWS"), (1, "FAKE NEWS")]:

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    samples = test_df[test_df["label"] == label].sample(
        2, random_state=42
    )

    for index, row in samples.iterrows():
        print("\nSource:", row["source_dataset"])
        print("News:", row["news"])
        print("Expected Label:", label)