import os

# Set environment variables to prevent OpenBLAS memory allocation errors
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import base64
import io
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

import skin_cancer_detection as SCD

app = Flask(__name__)

@app.context_processor
def inject_now():
    return {'now': datetime.now()}



CLASS_DETAILS = {
    "Actinic keratoses": {
        "name": "Actinic keratosis",
        "risk": "High priority",
        "risk_level": "medium",
        "description": (
            "A rough, sun-damaged patch that can develop into squamous cell "
            "carcinoma if it is not treated."
        ),
    },
    "Basal cell carcinoma": {
        "name": "Basal cell carcinoma",
        "risk": "Cancerous",
        "risk_level": "high",
        "description": (
            "A common skin cancer that often appears as a pearly bump or sore, "
            "usually on sun-exposed skin."
        ),
    },
    "Benign keratosis-like lesions": {
        "name": "Benign keratosis-like lesion",
        "risk": "Usually non-cancerous",
        "risk_level": "low",
        "description": (
            "A benign lesion that can still look suspicious in photos and should "
            "be confirmed clinically."
        ),
    },
    "Dermatofibroma": {
        "name": "Dermatofibroma",
        "risk": "Usually non-cancerous",
        "risk_level": "low",
        "description": (
            "A firm benign skin nodule that commonly appears on the arms or legs."
        ),
    },
    "Melanocytic nevi": {
        "name": "Melanocytic nevus",
        "risk": "Usually non-cancerous",
        "risk_level": "low",
        "description": "A common mole formed by pigment-producing skin cells.",
    },
    "Melanoma": {
        "name": "Melanoma",
        "risk": "Cancerous",
        "risk_level": "high",
        "description": (
            "A serious form of skin cancer that needs urgent medical evaluation."
        ),
    },
    "Vascular lesions": {
        "name": "Vascular lesion",
        "risk": "Needs review",
        "risk_level": "low",
        "description": (
            "A blood-vessel-related lesion that can sometimes bleed or mimic other "
            "skin findings."
        ),
    },
}


def load_preview_image(image_stream):
    image = Image.open(image_stream).convert("RGB")
    return image.copy()


def image_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/platform", methods=["GET"])
def platform():
    return render_template("platform.html")


@app.route("/workflow", methods=["GET"])
def workflow():
    return render_template("workflow.html")


@app.route("/faq", methods=["GET"])
def faq():
    return render_template("faq.html")


@app.route("/showresult", methods=["POST"])
def show_result():
    uploaded_file = request.files.get("pic")
    if not uploaded_file or not uploaded_file.filename:
        return render_template(
            "home.html",
            error="Please choose an image file before submitting.",
        )

    try:
        preview_image = load_preview_image(uploaded_file.stream)
    except (UnidentifiedImageError, OSError):
        return render_template(
            "home.html",
            error="That file could not be read as an image. Please upload JPG or PNG.",
        )

    # Perform inference
    probabilities, top_index, confidence = SCD.predict(preview_image)
    predicted_label = SCD.CLASS_NAMES[top_index]
    diagnosis = CLASS_DETAILS[predicted_label]

    # Generate Grad-CAM Heatmap
    original_img, heatmap_img = SCD.get_heatmap_overlay(preview_image, top_index)

    # Build metadata for future integration
    metadata = {
        "filename": uploaded_file.filename,
        "prediction": diagnosis["name"],
        "confidence": round(confidence * 100, 2),
        "risk_level": diagnosis["risk_level"],
        "body_location": "Placeholder",
        "timestamp": datetime.now().isoformat()
    }

    ranked_predictions = []
    for index, score in sorted(
        enumerate(probabilities), key=lambda item: item[1], reverse=True
    ):
        class_label = SCD.CLASS_NAMES[index]
        class_details = CLASS_DETAILS[class_label]
        ranked_predictions.append(
            {
                "name": class_details["name"],
                "risk": class_details["risk"],
                "probability": round(float(score) * 100, 2),
            }
        )

    return render_template(
        "results.html",
        image_data=image_to_data_url(original_img),
        heatmap_data=image_to_data_url(heatmap_img),
        diagnosis=diagnosis,
        metadata=metadata,
        ranked_predictions=ranked_predictions,
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
