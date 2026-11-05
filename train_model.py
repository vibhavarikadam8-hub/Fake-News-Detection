
import os
import re
import json
import joblib
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. PROJECT PATHS
# ==========================================

DATASET_PATH = os.path.join(
    "dataset", "WELFake_Dataset.csv"
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n================================")
print("     FAKE NEWS DETECTION")
print("================================")

# ==========================================
# 2. LOAD DATASET
# ==========================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Check the CSV filename inside the dataset folder."
    )

df = pd.read_csv(DATASET_PATH)
print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nExamples of Label 0:")
print(df[df["label"] == 0][["title", "text"]].head(5).to_string())

print("\nExamples of Label 1:")
print(df[df["label"] == 1][["title", "text"]].head(5).to_string())

print("\nDataset loaded successfully!")
print("Original dataset shape:", df.shape)
print("Available columns:", list(df.columns))

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

required_columns = ["label", "text"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing. "
            f"Found columns: {list(df.columns)}"
        )

# ==========================================
# 3. DATA CLEANING
# ==========================================

# Remove rows without labels
df = df.dropna(subset=["label"])

# Clean labels
df["label"] = pd.to_numeric(
    df["label"], errors="coerce"
)

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# WELFake convention:
# 0 = Fake News
# 1 = Real News

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Dataset labels must be 0 and 1.")

# Combine title and article text when title exists
if "title" in df.columns:
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    df["news"] = (
        df["title"].str.strip()
        + " "
        + df["text"].str.strip()
    )
else:
    df["news"] = df["text"].fillna("").astype(str)

# Remove empty articles
df["news"] = df["news"].str.strip()
df = df[df["news"].str.len() > 20]

# Remove exact duplicate articles before splitting
df["duplicate_key"] = (
    df["news"].str.lower().str.replace(
        r"\s+", " ", regex=True
    )
)

df = df.drop_duplicates(
    subset="duplicate_key"
).copy()

df = df.drop(columns=["duplicate_key"])

print("\nCleaned dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())
df = df.reset_index(drop=True)

# ==========================================
# 4. PREPARE INPUT AND OUTPUT
# ==========================================

X = df["news"]
y = df["label"]
print("\nLABEL DISTRIBUTION:")
print(df["label"].value_counts())

print("\nSAMPLE NEWS WITH LABELS:")
print(df[["news", "label"]].head(10))

# Stratified split preserves class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 5. TF-IDF VECTORIZATION
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words="english",
    sublinear_tf=True,
    strip_accents="unicode"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF features:", X_train_tfidf.shape[1])

# ==========================================
# 6. MACHINE LEARNING MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=2.0,
        random_state=42
    ),

    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=1000,
        C=0.5,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=0.1
    )
}

results = []
best_model = None
best_model_name = None
best_f1 = -1

# ==========================================
# 7. TRAIN AND EVALUATE
# ==========================================

for name, model in models.items():

    print("\n--------------------------------")
    print("Training:", name)
    print("--------------------------------")

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

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

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    # ROC-AUC using model scores
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(
                X_test_tfidf
            )[:, 1]
        else:
            scores = model.decision_function(
                X_test_tfidf
            )

        auc = roc_auc_score(y_test, scores)

    except Exception:
        auc = None

    print("ROC-AUC:", auc)

    results.append({
        "model": name,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(auc, 4) if auc is not None else None
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "best_model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

# ==========================================
# 9. SAVE EVALUATION RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False
)

results_df.to_csv(
    os.path.join(MODEL_DIR, "model_results.csv"),
    index=False
)

metrics = {
    "best_model": best_model_name,
    "accuracy": round(
        accuracy_score(
            y_test, best_model.predict(X_test_tfidf)
        ) * 100, 2
    ),
    "precision": round(
        precision_score(
            y_test,
            best_model.predict(X_test_tfidf),
            zero_division=0
        ) * 100, 2
    ),
    "recall": round(
        recall_score(
            y_test,
            best_model.predict(X_test_tfidf),
            zero_division=0
        ) * 100, 2
    ),
    "f1_score": round(best_f1 * 100, 2),
    "confusion_matrix": confusion_matrix(
        y_test,
        best_model.predict(X_test_tfidf)
    ).tolist()
}

with open(
    os.path.join(MODEL_DIR, "metrics.json"),
    "w"
) as file:
    json.dump(metrics, file, indent=4)

# ==========================================
# 10. FINAL REPORT
# ==========================================

print("\n================================")
print("        MODEL COMPARISON")
print("================================")

print(results_df.to_string(index=False))

print("\nBest model:", best_model_name)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    best_model.predict(X_test_tfidf)
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    best_model.predict(X_test_tfidf),
    target_names=["REAL NEWS", "FAKE NEWS"],
    zero_division=0
))

print("\n================================")
print("       TRAINING COMPLETED")
print("================================")

print("\nSaved files:")
print("models/best_model.pkl")
print("models/tfidf_vectorizer.pkl")
print("models/model_results.csv")
print("models/metrics.json")

print("\nYour ML model is ready for Flask integration!")
