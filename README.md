# 🩺 MelanoDetect: Skin Cancer Classification

MelanoDetect is a deep-learning-powered web application designed to assist in the early detection and classification of skin lesions. By leveraging state-of-the-art computer vision models, it provides users with instantaneous feedback on skin concerns, categorized into seven distinct clinical classes.

## 🚀 Key Features

- **Instant Classification**: Upload an image of a skin lesion and receive a diagnosis in seconds.
- **Multi-Class Analysis**: Identifies 7 types of skin lesions, including Melanoma, Basal Cell Carcinoma, and more.
- **Confidence Ranking**: Provides a detailed breakdown of prediction confidence for the top diagnoses.
- **Clinical Context**: Includes descriptions and risk levels for each identified condition.
- **Responsive Web Interface**: A modern, easy-to-use platform compatible with desktop and mobile browsing.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Deep Learning**: PyTorch, Torchvision
- **Architecture**: MobileNetV2 (optimized for speed and efficiency)
- **Frontend**: HTML5, CSS3 (Vanilla JS for interactive elements)
- **Data Source**: HAM10000 Dataset

## 🧠 How It Works

The system operates through a specialized AI pipeline:

1.  **Image Acquisition**: The user uploads a high-resolution image of a skin lesion.
2.  **Digital Preprocessing**:
    *   The image is converted to the **RGB** color space.
    *   It is resized to **224x224** pixels to match the model's input layer.
    *   Pixel values are normalized using the dataset-specific mean (`[0.763, 0.546, 0.570]`) and standard deviation (`[0.141, 0.153, 0.170]`).
3.  **Neural Inference**:
    *   The preprocessed tensor is fed into a **MobileNetV2** model.
    *   The model extract deep spatial features and outputs raw scores (logits) for the 7 classes.
    *   A **Softmax** function converts these scores into probability percentages.
4.  **Information Mapping**:
    *   The system maps the highest-probability class to its clinical name, risk factor, and descriptive summary.
    *   Results are rendered dynamically on the `results.html` page.

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/VedantRaut1/MelanoDetect.git
cd MelanoDetect
```

### 2. Set Up a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

## 📈 Future Enhancements

- [ ] **Explainable AI (XAI)**: Implement Grad-CAM to highlight exactly where the model is looking when making a prediction.
- [ ] **Confidence Thresholds**: Alert users if the model's confidence is below a certain threshold to recommend professional consultation.
- [ ] **Mobile App Integration**: Develop a native mobile app for faster image capture and analysis.
- [ ] **User Feedback Loop**: Allow users to upload confirmed medical reports to help fine-tune the model for higher accuracy.

## ⚠️ Disclaimer

**This tool is for educational and screening purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.
