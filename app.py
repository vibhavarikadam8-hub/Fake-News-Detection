import os
import json
import math
import joblib
import pandas as pd

from flask import Flask, render_template, request, jsonify

# =====================================
# FLASK APPLICATION
# =====================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# New Combined Model Paths
MODEL_PATH = os.path.join(
    MODEL_DIR, "combined_best_model_candidate.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR, "combined_tfidf_vectorizer_candidate.pkl"
)
METRICS_PATH = os.path.join(
    MODEL_DIR, "combined_metrics.json"
)

# =====================================
# LOAD TRAINED MODEL
# =====================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Combined trained model not found!"
    )

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        "Combined TF-IDF vectorizer not found!"
    )

model = joblib.load(MODEL_PATH)

vectorizer = joblib.load(VECTORIZER_PATH)

print("\n================================")
print("MODEL LOADING")
print("================================")

print("Model:", type(model).__name__)
print("Model classes:", model.classes_)

if set(model.classes_) != {0, 1}:
    raise ValueError("Unexpected model label classes!")

print("ML model loaded successfully!")
print("TF-IDF vectorizer loaded successfully!")

# =====================================
# HOME PAGE
# =====================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================
# NEWS PREDICTION API
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True) or {}

        news_text = str(
            data.get("news", "")
        ).strip()

        # Input validation
        if not news_text:

            return jsonify({
                "error": "Please enter a news article."
            }), 400

        if len(news_text) < 20:

            return jsonify({
                "error": "Please enter at least 20 characters."
            }), 400

        if len(news_text) > 100000:

            return jsonify({
                "error": "News text is too long."
            }), 400

        # =====================================
        # TF-IDF TRANSFORMATION
        # =====================================

        transformed_text = vectorizer.transform(
            [news_text]
        )

        # =====================================
        # MODEL PREDICTION
        # =====================================

        prediction = int(
            model.predict(transformed_text)[0]
        )

        print("\n================================")
        print("NEWS PREDICTION")
        print("================================")

        print("INPUT NEWS:", news_text)
        print("MODEL:", type(model).__name__)
        print("RAW PREDICTION:", prediction)

        # Combined Dataset Mapping
        # 0 = REAL NEWS
        # 1 = FAKE NEWS

        if prediction == 0:

            result = "REAL NEWS"

        elif prediction == 1:

            result = "FAKE NEWS"

        else:

            return jsonify({
                "error": "Unexpected model prediction."
            }), 500

        print("FINAL RESULT:", result)

        # =====================================
        # DECISION MARGIN SCORE
        # =====================================

        score = None
        score_type = None

        if hasattr(model, "decision_function"):

            decision = float(
                model.decision_function(
                    transformed_text
                )[0]
            )

            margin = abs(decision)

            score = (
                100 / (
                    1 + math.exp(-min(margin, 50))
                )
            )

            score_type = "decision_margin_score"

        # =====================================
        # EXPLAIN INFLUENTIAL WORDS
        # =====================================

        explanation = []

        if hasattr(model, "coef_"):

            coefficients = model.coef_[0]

            feature_names = (
                vectorizer.get_feature_names_out()
            )

            feature_values = (
                transformed_text.toarray()[0]
            )

            contributions = (
                feature_values * coefficients
            )

            if prediction == 1:

                top_indices = (
                    contributions.argsort()[-5:][::-1]
                )

            else:

                top_indices = (
                    contributions.argsort()[:5]
                )

            for index in top_indices:

                if feature_values[index] > 0:

                    explanation.append({

                        "word": str(
                            feature_names[index]
                        ),

                        "contribution": round(
                            float(contributions[index]), 5
                        )

                    })

        # =====================================
        # RETURN PREDICTION
        # =====================================

        return jsonify({

            "success": True,

            "prediction": result,

            "label": prediction,

            "score": (
                round(score, 2)
                if score is not None
                else None
            ),

            "score_type": score_type,

            "explanation": explanation,

            "message": (
                "This is an ML model prediction, "
                "not independent verification of factual claims."
            )

        })

    except Exception:

        app.logger.exception(
            "Prediction failed"
        )

        return jsonify({

            "error": (
                "Prediction failed. Please try again."
            )

        }), 500


# =====================================
# MODEL METRICS API
# =====================================

@app.route("/api/metrics")
def get_metrics():

    try:

        with open(
            METRICS_PATH, "r"
        ) as file:

            metrics = json.load(file)

        return jsonify(metrics)

    except Exception:

        app.logger.exception(
            "Metrics loading failed"
        )

        return jsonify({

            "error": "Model metrics are unavailable."

        }), 500


# =====================================
# MODEL COMPARISON API
# =====================================

@app.route("/api/models")
def get_models():

    try:

        if not os.path.exists(METRICS_PATH):

            return jsonify({
                "error": "Model metrics are unavailable."
            }), 404

        with open(
            METRICS_PATH, "r"
        ) as file:

            metrics = json.load(file)

        results = []

        for name, values in metrics.items():

            results.append({

                "model": name,

                "accuracy": values["accuracy"],

                "precision": values["precision"],

                "recall": values["recall"],

                "f1_score": values["f1_score"]

            })

        return jsonify(results)

    except Exception:

        app.logger.exception(
            "Model comparison failed"
        )

        return jsonify({

            "error": (
                "Model comparison data unavailable."
            )

        }), 500


# =====================================
# RUN APPLICATION
# =====================================

if __name__ == "__main__":

    print("\n================================")
    print("FAKE NEWS DETECTION SYSTEM")
    print("================================")

    print("Selected Model: Linear SVM")
    print("Training Dataset: WELFake + IFND")

    print("Label 0: REAL NEWS")
    print("Label 1: FAKE NEWS")

    print("Open: http://127.0.0.1:5000")

    print("================================")

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )

