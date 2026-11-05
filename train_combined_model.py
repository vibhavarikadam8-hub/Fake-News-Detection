import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# 1. LOAD DATASET
# -----------------------------------

df = pd.read_csv("dataset/combined_news.csv")

print("Dataset loaded:", df.shape)

X = df["news"].astype(str)
y = df["label"].astype(int)

# -----------------------------------
# 2. SPLIT EACH DATASET SEPARATELY
# -----------------------------------

train_parts = []
test_parts = []

for source in df["source_dataset"].unique():

    source_data = df[df["source_dataset"] == source]

    train_data, test_data = train_test_split(
        source_data,
        test_size=0.20,
        random_state=42,
        stratify=source_data["label"]
    )

    train_parts.append(train_data)
    test_parts.append(test_data)

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

print("\nTraining records:", len(train_df))
print("Testing records:", len(test_df))

# -----------------------------------
# 3. TF-IDF VECTORIZATION
# -----------------------------------

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words="english",
    sublinear_tf=True,
    strip_accents="unicode"
)

X_train = vectorizer.fit_transform(train_df["news"])
X_test = vectorizer.transform(test_df["news"])

y_train = train_df["label"]
y_test = test_df["label"]

print("\nTF-IDF completed!")
print("Feature count:", len(vectorizer.get_feature_names_out()))

# -----------------------------------
# 4. DEFINE ML MODELS
# -----------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=2.0,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=0.1
    ),

    "Passive Aggressive": SGDClassifier(
        loss="hinge",
        penalty=None,
        learning_rate="pa1",
        eta0=1.0,
        random_state=42
    )
}

# -----------------------------------
# 5. TRAIN AND EVALUATE
# -----------------------------------

results = {}
best_model = None
best_model_name = ""
best_f1 = -1

os.makedirs("models", exist_ok=True)

for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["REAL NEWS", "FAKE NEWS"],
        zero_division=0
    ))

    # Evaluate separately on both datasets
    source_results = {}

    for source in test_df["source_dataset"].unique():

        mask = test_df["source_dataset"] == source

        source_results[source] = {
            "accuracy": round(
                accuracy_score(
                    y_test[mask],
                    predictions[mask]
                ) * 100, 2
            ),
            "f1_score": round(
                f1_score(
                    y_test[mask],
                    predictions[mask],
                    zero_division=0
                ) * 100, 2
            )
        }

    print("\nDataset-wise performance:")
    print(source_results)

    results[name] = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "dataset_results": source_results
    }

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# -----------------------------------
# 6. SAVE BEST MODEL
# -----------------------------------

joblib.dump(
    best_model,
    "models/combined_best_model_candidate.pkl"
)

joblib.dump(
    vectorizer,
    "models/combined_tfidf_vectorizer_candidate.pkl"
)

with open(
    "models/combined_metrics.json",
    "w"
) as file:

    json.dump(results, file, indent=4)

print("\n================================")
print("FINAL MODEL COMPARISON")
print("================================")

for name, metrics in results.items():
    print(name, ":", metrics)

print("\nBest Model:", best_model_name)
print("Best F1 Score:", round(best_f1 * 100, 2), "%")

print("\nModel and vectorizer saved successfully!")