from flask import Flask, render_template, request, jsonify
import joblib
import re
import os
import numpy as np
import pandas as pd


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

# app.py is inside:
# AI-Phishing-Email-Detector/app/
#
# The models folder is:
# AI-Phishing-Email-Detector/models/

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_phishing_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "models",
    "metadata_features.pkl"
)

STOPWORDS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "stop_words.pkl"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading phishing detection model...")

try:

    model = joblib.load(MODEL_PATH)

    print(
        "Loaded model:",
        type(model).__name__
    )

except Exception as e:

    print("ERROR loading model:")
    print(e)

    model = None


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

try:

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    print(
        "Loaded TF-IDF vectorizer"
    )

except Exception as e:

    print("ERROR loading TF-IDF vectorizer:")
    print(e)

    vectorizer = None


# ============================================================
# LOAD METADATA FEATURES
# ============================================================

try:

    metadata_features = joblib.load(
        METADATA_PATH
    )

    print(
        "Metadata features:",
        metadata_features
    )

except Exception as e:

    print("ERROR loading metadata features:")
    print(e)

    metadata_features = []


# ============================================================
# LOAD STOP WORDS
# ============================================================

try:

    stop_words = joblib.load(
        STOPWORDS_PATH
    )

    print(
        "Stop words loaded successfully"
    )

except Exception as e:

    print("WARNING: Could not load stop words")

    stop_words = set()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not isinstance(text, str):

        text = str(text)

    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Keep alphabetic characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Remove stopwords if available
    if stop_words:

        words = text.split()

        words = [
            word
            for word in words
            if word not in stop_words
        ]

        text = " ".join(words)

    return text


# ============================================================
# METADATA EXTRACTION
# ============================================================

def extract_metadata(email_text):

    email_text = str(email_text)

    metadata = {}

    # Email length
    metadata["email_length"] = len(
        email_text
    )

    # URL count
    metadata["url_count"] = len(
        re.findall(
            r"https?://\S+|www\.\S+",
            email_text,
            flags=re.IGNORECASE
        )
    )

    # Number of digits
    metadata["digit_count"] = len(
        re.findall(
            r"\d",
            email_text
        )
    )

    # Special characters
    metadata["special_char_count"] = len(
        re.findall(
            r"[^a-zA-Z0-9\s]",
            email_text
        )
    )

    # Exclamation marks
    metadata["exclamation_count"] = email_text.count(
        "!"
    )

    # Suspicious words
    suspicious_words = [
        "urgent",
        "verify",
        "verification",
        "password",
        "login",
        "account",
        "suspended",
        "suspension",
        "click",
        "confirm",
        "security",
        "bank",
        "payment",
        "credential",
        "credentials",
        "immediately",
        "winner",
        "prize",
        "reward",
        "refund",
        "invoice",
        "expire",
        "expired",
        "locked",
        "unlock",
        "limited",
        "warning",
        "alert",
        "claim"
    ]

    lower_text = email_text.lower()

    metadata["suspicious_word_count"] = sum(
        lower_text.count(word)
        for word in suspicious_words
    )

    return metadata


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_features(email_text):

    # Clean email
    cleaned_email = clean_text(
        email_text
    )

    # TF-IDF
    tfidf_features = vectorizer.transform(
        [cleaned_email]
    )

    # Extract metadata
    metadata = extract_metadata(
        email_text
    )

    # --------------------------------------------------------
    # Create metadata vector
    # --------------------------------------------------------

    metadata_vector = []

    for feature in metadata_features:

        metadata_vector.append(
            metadata.get(
                feature,
                0
            )
        )

    metadata_vector = np.array(
        metadata_vector,
        dtype=float
    ).reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # Combine TF-IDF + metadata
    # --------------------------------------------------------

    from scipy.sparse import csr_matrix
    from scipy.sparse import hstack

    metadata_sparse = csr_matrix(
        metadata_vector
    )

    combined_features = hstack(
        [
            tfidf_features,
            metadata_sparse
        ]
    )

    return combined_features, metadata


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if model is None:

            return jsonify({
                "error":
                "Model could not be loaded."
            }), 500


        if vectorizer is None:

            return jsonify({
                "error":
                "TF-IDF vectorizer could not be loaded."
            }), 500


        # ----------------------------------------------------
        # Get JSON data
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "error":
                "No JSON data received."
            }), 400


        email_text = data.get(
            "email",
            ""
        )


        # ----------------------------------------------------
        # Validate email
        # ----------------------------------------------------

        if not email_text:

            return jsonify({
                "error":
                "Please enter an email."
            }), 400


        if not isinstance(
            email_text,
            str
        ):

            email_text = str(
                email_text
            )


        # ----------------------------------------------------
        # Create features
        # ----------------------------------------------------

        features, metadata = create_features(
            email_text
        )


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model.predict(
            features
        )[0]


        # ----------------------------------------------------
        # Get confidence
        # ----------------------------------------------------

        confidence = 0.0

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                features
            )[0]

            confidence = float(
                np.max(probabilities)
                * 100
            )


        # ----------------------------------------------------
        # Convert prediction to label
        # ----------------------------------------------------

        prediction_string = str(
            prediction
        ).lower()


        if (
            prediction_string in
            [
                "1",
                "phishing",
                "spam",
                "true",
                "malicious"
            ]
        ):

            label = "PHISHING"

        elif (
            prediction_string in
            [
                "0",
                "legitimate",
                "ham",
                "false",
                "safe"
            ]
        ):

            label = "LEGITIMATE"

        else:

            # Handle numeric predictions
            try:

                numeric_prediction = int(
                    prediction
                )

                if numeric_prediction == 1:

                    label = "PHISHING"

                else:

                    label = "LEGITIMATE"

            except:

                label = str(
                    prediction
                ).upper()


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        result = {

            "label": label,

            "confidence": round(
                confidence,
                2
            ),

            "metadata": metadata

        }


        return jsonify(
            result
        )


    except Exception as e:

        print(
            "\n================================="
        )

        print(
            "PREDICTION ERROR"
        )

        print(
            "================================="
        )

        print(
            repr(e)
        )

        import traceback

        traceback.print_exc()


        return jsonify({

            "error":
            "Prediction failed.",

            "details":
            str(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "running",

        "model_loaded":
        model is not None,

        "vectorizer_loaded":
        vectorizer is not None,

        "metadata_features":
        metadata_features

    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "AI PHISHING EMAIL DETECTOR"
    )

    print(
        "========================================"
    )

    print(
        "Model:",
        type(model).__name__
        if model is not None
        else "NOT LOADED"
    )

    print(
        "TF-IDF:",
        "Loaded"
        if vectorizer is not None
        else "NOT LOADED"
    )

    print(
        "Metadata:",
        metadata_features
    )

    print(
        "========================================"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )