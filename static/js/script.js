
// ======================================
// TRUTHLENS - FAKE NEWS DETECTION
// FRONTEND JAVASCRIPT
// ======================================

document.addEventListener("DOMContentLoaded", () => {

    const newsInput = document.getElementById("newsInput");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const clearBtn = document.getElementById("clearBtn");

    const loading = document.getElementById("loading");
    const resultBox = document.getElementById("resultBox");

    const charCount = document.getElementById("charCount");

    // ==================================
    // CHARACTER COUNTER
    // ==================================

    newsInput.addEventListener("input", () => {
        charCount.textContent =
            `${newsInput.value.length} characters`;
    });

    // ==================================
    // ANALYZE NEWS
    // ==================================

    analyzeBtn.addEventListener("click", analyzeNews);

    async function analyzeNews() {

        const news = newsInput.value.trim();

        if (news.length < 20) {
            alert("Please enter at least 20 characters of news.");
            newsInput.focus();
            return;
        }

        // Show loading animation
        resultBox.classList.add("hidden");
        loading.classList.remove("hidden");

        analyzeBtn.disabled = true;

        document.getElementById("buttonText").textContent =
            "Analyzing...";

        try {

            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    news: news
                })

            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error || "Prediction failed."
                );
            }

            if (!data.success) {
                throw new Error("Unable to analyze the news.");
            }

            displayResult(data);

        } catch (error) {

            alert(error.message);

        } finally {

            loading.classList.add("hidden");

            analyzeBtn.disabled = false;

            document.getElementById("buttonText").textContent =
                "Analyze News →";

        }

    }

    // ==================================
    // DISPLAY PREDICTION
    // ==================================

    function displayResult(data) {

        const predictionText =
            document.getElementById("predictionText");

        const resultIcon =
            document.getElementById("resultIcon");

        const scoreValue =
            document.getElementById("scoreValue");

        const scoreProgress =
            document.getElementById("scoreProgress");

        const scoreDescription =
            document.getElementById("scoreDescription");

        const scoreLabel =
            document.getElementById("scoreLabel");

        const wordList =
            document.getElementById("wordList");

        resultBox.classList.remove("hidden", "fake", "real");

        const isFake = data.prediction === "FAKE NEWS";

        if (isFake) {

            resultBox.classList.add("fake");

            predictionText.textContent = "FAKE NEWS";

            resultIcon.textContent = "⚠";

        } else {

            resultBox.classList.add("real");

            predictionText.textContent = "REAL NEWS";

            resultIcon.textContent = "✓";

        }

        // Display score only when supplied
        if (data.score !== null && data.score !== undefined) {

            const score = Math.max(
                0,
                Math.min(100, Number(data.score))
            );

            scoreValue.textContent =
                `${score.toFixed(2)}%`;

            scoreProgress.style.width =
                `${score}%`;

            if (data.score_type === "model_probability") {

                scoreLabel.textContent =
                    "Model Probability";

                scoreDescription.textContent =
                    "Estimated probability assigned by the classifier. It may not be calibrated.";

            } else {

                scoreLabel.textContent =
                    "Decision Margin Score";

                scoreDescription.textContent =
                    "A transformed model decision margin, not a probability or verified truth score.";

            }

        } else {

            scoreValue.textContent = "Unavailable";
            scoreProgress.style.width = "0%";

            scoreDescription.textContent =
                "The selected model does not provide a confidence score.";

        }

        // Display important words safely
        wordList.replaceChildren();

        if (
            Array.isArray(data.explanation) &&
            data.explanation.length > 0
        ) {

            data.explanation.forEach(item => {

                const chip = document.createElement("span");

                chip.className = "word-chip";

                chip.textContent = item.word;

                wordList.appendChild(chip);

            });

        } else {

            wordList.textContent =
                "No influential words available for this model.";

        }

        // Scroll to results
        resultBox.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });

    }

    // ==================================
    // CLEAR BUTTON
    // ==================================

    clearBtn.addEventListener("click", () => {

        newsInput.value = "";

        charCount.textContent = "0 characters";

        resultBox.classList.add("hidden");

        loading.classList.add("hidden");

        newsInput.focus();

    });

    // ==================================
    // LOAD MODEL METRICS
    // ==================================

    async function loadMetrics() {

        const status =
            document.getElementById("metricsStatus");

        try {

            const response = await fetch("/api/metrics");

            if (!response.ok) {
                throw new Error("Metrics unavailable");
            }

            const metrics = await response.json();

            if (metrics.error) {
                throw new Error(metrics.error);
            }

            document.getElementById("stat-model").textContent =
                metrics.best_model || "Unavailable";

            document.getElementById("stat-accuracy").textContent =
                `${metrics.accuracy}%`;

            document.getElementById("metricAccuracy").textContent =
                `${metrics.accuracy}%`;

            document.getElementById("metricPrecision").textContent =
                `${metrics.precision}%`;

            document.getElementById("metricRecall").textContent =
                `${metrics.recall}%`;

            document.getElementById("metricF1").textContent =
                `${metrics.f1_score}%`;

            document.getElementById("bestModelName").textContent =
                metrics.best_model || "Unavailable";

            // Confusion matrix
            if (
                Array.isArray(metrics.confusion_matrix) &&
                metrics.confusion_matrix.length === 2 &&
                metrics.confusion_matrix.every(
                    row => Array.isArray(row) && row.length === 2
                )
            ) {

                document.getElementById("cm00").textContent =
                    metrics.confusion_matrix[0][0];

                document.getElementById("cm01").textContent =
                    metrics.confusion_matrix[0][1];

                document.getElementById("cm10").textContent =
                    metrics.confusion_matrix[1][0];

                document.getElementById("cm11").textContent =
                    metrics.confusion_matrix[1][1];

            }

            status.textContent =
                "Successfully loaded actual test-set evaluation results.";

        } catch (error) {

            status.textContent =
                "Unable to load metrics: " + error.message;

        }

    }

    // ==================================
    // LOAD MODEL COMPARISON
    // ==================================

    async function loadModelComparison() {

        const container =
            document.getElementById("modelComparison");

        try {

            const response = await fetch("/api/models");

            if (!response.ok) {
                throw new Error("Model comparison unavailable");
            }

            const models = await response.json();

            if (!Array.isArray(models) || models.length === 0) {
                throw new Error("No model results found.");
            }

            container.replaceChildren();

            models.forEach(model => {

                const item =
                    document.createElement("div");

                item.className = "comparison-item";

                const info =
                    document.createElement("div");

                info.className = "comparison-info";

                const name =
                    document.createElement("span");

                name.textContent = model.model;

                const score =
                    document.createElement("span");

                const f1 = Number(model.f1_score);

                score.textContent =
                    Number.isFinite(f1) ? `${f1.toFixed(2)}% F1` : "N/A";

                const track =
                    document.createElement("div");

                track.className = "comparison-track";

                const fill =
                    document.createElement("div");

                fill.className = "comparison-fill";

                fill.style.width =
                    `${Math.max(0, Math.min(100, f1 || 0))}%`;

                info.append(name, score);

                track.appendChild(fill);

                item.append(info, track);

                container.appendChild(item);

            });

        } catch (error) {

            container.textContent =
                "Unable to load model comparison: " + error.message;

        }

    }

    // ==================================
    // INITIALIZE DASHBOARD
    // ==================================

    loadMetrics();

    loadModelComparison();

});
