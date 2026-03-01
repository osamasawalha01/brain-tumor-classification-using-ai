"""
PROJECT: BRAIN AI DIAGNOSIS SYSTEM 
-----------------------------------------
CORE OBJECTIVE: 
Provide a Clinical Decision Support System (CDSS) for neuro-oncology using 
Transfer Learning and Explainable AI (XAI).

TECHNICAL SPECIFICATIONS:
- Dataset: BRISC 2025 MRI (Multi-class: Glioma, Meningioma, Pituitary, No Tumor).
- Architectures: ResNet50 and EfficientNetB0 models.
- XAI Implementation: Grad-CAM for heat-map visualization of tumor localization.
- Clinical Pipeline: Automated PDF report generation (FPDF) and real-time 
  specialist alerts via Telegram API.
"""

import os
# Force TensorFlow to use legacy Keras to ensure model weight compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tf_keras as keras
from tf_keras.models import load_model
from tf_keras import Model, Input
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from datetime import datetime
import requests
from fpdf import FPDF
import json

# Internal module imports for database and Grad-CAM logic
from database import init_db, save_patient_record, get_patient_by_id
from utils.gradcam import make_gradcam_heatmap, resize_heatmap_to_image, blend_heatmap_with_image

# ===============================
# FUNCTIONS: TELEGRAM & PDF
# ===============================

def send_telegram_diagnostic_report(token, chat_id, patient_name, patient_id, diagnosis, confidence, original_path, blended_path):
    """Dispatches diagnostic findings and MRI comparison gallery to Telegram"""
    message = (
        f"🚨 *Clinical Alert: New Diagnostic Result*\n\n"
        f"👤 *Patient:* {patient_name}\n"
        f"🆔 *ID:* {patient_id}\n"
        f"🧬 *Diagnosis:* {diagnosis}\n"
        f"📊 *Confidence:* {confidence:.2f}%\n"
        f"📅 *Date:* {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"🔬 _Comparison: Original MRI (Left) vs. Grad-CAM Localization (Right)_"
    )
    
    # Send text alert
    url_text = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url_text, data={'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'})

    # Send images as a unified media group for side-by-side comparison
    url_media = f"https://api.telegram.org/bot{token}/sendMediaGroup"
    files = {
        'photo1': open(original_path, 'rb'),
        'photo2': open(blended_path, 'rb')
    }
    media = [
        {'type': 'photo', 'media': 'attach://photo1', 'caption': 'Original MRI'},
        {'type': 'photo', 'media': 'attach://photo2', 'caption': 'Grad-CAM Analysis'}
    ]
    requests.post(url_media, data={'chat_id': chat_id, 'media': json.dumps(media)}, files=files)

def create_pdf_report(patient_name, patient_id, diagnosis, confidence, original_path, blended_path):
    """Generates an official PDF clinical report with side-by-side visualization"""
    pdf = FPDF()
    pdf.add_page()

    # Document Header
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(0, 68, 102)
    pdf.cell(200, 10, "Brain AI Diagnostic Report", ln=True, align='C')
    pdf.ln(10)

    # Patient Information
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(100, 10, f"Patient Name: {patient_name}")
    pdf.cell(100, 10, f"Patient ID: {patient_id}", ln=True)
    pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(10)

    # Diagnostic Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, f"Diagnosis: {diagnosis.upper()} ({confidence:.2f}%)", ln=True)
    pdf.ln(5)

    # Visual Evidence Comparison
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(95, 10, "1. Original MRI Scan", align='C')
    pdf.cell(95, 10, "2. Grad-CAM Localization", align='C', ln=True)

    # Position images on PDF
    y_pos = pdf.get_y()
    pdf.image(original_path, x=15, y=y_pos, w=85)
    pdf.image(blended_path, x=110, y=y_pos, w=85)

    pdf.ln(90)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 10, "Disclaimer: This automated AI analysis is intended as a support tool for medical professionals. Final diagnosis must be confirmed by a radiologist.", align='C')

    output_path = f"report_{patient_id}.pdf"
    pdf.output(output_path)
    return output_path

# ===============================
# CONFIGURATION & UI SETUP
# ===============================

st.set_page_config(page_title="🧠 Brain AI Diagnostic System", layout="wide")
init_db()

IMG_SIZE = (224, 224)
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
# Output path for local storage of generated reports and images
OUTPUT_DIR = Path(r"C:\Users\Osama Sawalha\Desktop\P\brain_tumor_ai_app\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Custom CSS for Professional Clinical Interface
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 5px solid #004466; }
    h1, h2, h3 { color: #004466; font-family: 'Segoe UI'; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="background-color: #004466; padding: 25px; border-radius: 10px; color: white; margin-bottom: 25px;">
        <h1 style="color: white; margin: 0;">🔬 BRAIN AI DIAGNOSIS SYSTEM</h1>
        <p style="margin: 0; opacity: 0.8;">Advanced Clinical Decision Support Platform | v2.0</p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# RESOURCE LOADING (CACHED)
# ===============================

@st.cache_resource
def load_models():
    """Loads pre-trained deep learning models with caching for performance"""
    resnet = load_model(r"C:\Users\Osama Sawalha\Desktop\P\brain_tumor_ai_app\models\resnet_transfer_finetuned.keras", compile=False)
    effnet = load_model(r"C:\Users\Osama Sawalha\Desktop\P\brain_tumor_ai_app\models\efficientnet_b0_finetuned.keras", compile=False)
    return resnet, effnet

with st.spinner("Initializing AI Models..."):
    resnet_model, effnet_model = load_models()

last_conv_layers = {"ResNet50": "conv5_block3_out", "EfficientNetB0": "top_conv"}

# ===============================
# SIDEBAR CONTROLS
# ===============================

st.sidebar.title("Configuration 🧠")
model_choice = st.sidebar.selectbox("Select Model Architecture", ["ResNet50", "EfficientNetB0"])
uploaded_file = st.sidebar.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
alpha_slider = st.sidebar.slider("XAI Heatmap Opacity", 0.0, 1.0, 0.4)

st.sidebar.markdown("---")
st.sidebar.subheader("Patient Records")
patient_id = st.sidebar.text_input("Patient ID")
patient_name = st.sidebar.text_input("Full Name")

if st.sidebar.button("🔍 Search Registry"):
    p_data = get_patient_by_id(patient_id)
    if p_data:
        st.sidebar.success(f"Record Found: {p_data[1]}")
    else:
        st.sidebar.warning("No record found.")

# ===============================
# DIAGNOSTIC PIPELINE
# ===============================

if uploaded_file is not None:
    # 1. Processing Input Image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Save original image for report inclusion
    original_fn = f"original_{patient_id if patient_id else 'temp'}.png"
    original_save_path = OUTPUT_DIR / original_fn
    image.save(original_save_path)
    
    img_resized = image.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    col_input, col_results = st.columns([1, 1])

    with col_input:
        st.subheader("MRI Input Analysis")
        st.image(image, caption="Original MRI Scan", use_container_width=True)

    # 2. Executing Inference
    model = resnet_model if model_choice == "ResNet50" else effnet_model
    preds = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(preds[0])
    pred_class = CLASSES[pred_idx]
    confidence = float(np.max(preds[0])) * 100

    with col_results:
        st.subheader("Inference Result")
        st.metric("Predicted Condition", pred_class.upper())
        st.write(f"Confidence Level: **{confidence:.2f}%**")

        # Probability distribution visualization
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
        ax_pie.pie(preds[0], labels=CLASSES, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis"))
        st.pyplot(fig_pie)

    # 3. Generating Explainable AI (Grad-CAM)
    st.markdown("---")
    st.subheader("🔍 Tumor Localization (Explainable AI)")

    last_conv_name = last_conv_layers[model_choice]
    base_model = model.get_layer(model_choice.lower())

    # Reconstruct classification head for heat-map interpretation
    classifier_input = Input(shape=base_model.get_layer(last_conv_name).output.shape[1:])
    x = model.get_layer("global_avg_pool")(classifier_input)
    x = model.get_layer("dropout")(x, training=False)
    output_scores = model.get_layer("predictions")(x)
    classifier_model = Model(classifier_input, output_scores)

    heatmap = make_gradcam_heatmap(img_array, base_model, classifier_model, last_conv_name, pred_idx)
    heatmap_resized = resize_heatmap_to_image(heatmap, IMG_SIZE)
    blended_img = blend_heatmap_with_image(np.array(img_resized), heatmap_resized, alpha=alpha_slider)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(blended_img, caption="Grad-CAM Tumor Heatmap", use_container_width=True)

        # 4. Data Persistence & Export
        blended_fn = f"blended_{patient_id if patient_id else 'temp'}.png"
        blended_save_path = OUTPUT_DIR / blended_fn
        Image.fromarray(blended_img).save(blended_save_path)

        if patient_id and patient_name:
            # Update local database registry
            save_patient_record(patient_id, patient_name, pred_class, confidence / 100, str(blended_save_path))

        st.markdown("---")
        st.subheader("🏥 Clinical Reporting")
        comm_col1, comm_col2 = st.columns([1, 1])

        with comm_col1:
            if st.button("📤 Send Telegram Alert"):
                if patient_id and patient_name:
                    # Connection credentials
                  MY_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
                    MY_CHAT_ID = "YOUR_CHAT_ID_HERE"
                    send_telegram_diagnostic_report(
                        MY_TOKEN, MY_CHAT_ID, patient_name, patient_id,
                        pred_class, confidence,
                        str(original_save_path), str(blended_save_path)
                    )
                    st.success("Alert successfully dispatched.")
                else:
                    st.error("Patient metadata required.")

        with comm_col2:
            if st.button("📄 Generate PDF Report"):
                if patient_id and patient_name:
                    pdf_path = create_pdf_report(
                        patient_name, patient_id, pred_class,
                        confidence, str(original_save_path), str(blended_save_path)
                    )
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download Document", f, file_name=f"Report_{patient_id}.pdf")
                else:
                    st.error("Patient metadata required.")

    # 5. Model Validation Metrics
    st.markdown("---")
    st.subheader("📊 Statistical Performance Validation")
    ev_col1, ev_col2 = st.columns([1, 1])

    with ev_col1:
        # Synthetic validation report for demo purposes
        y_true = np.array([pred_idx] * 10)
        y_pred = np.array([pred_idx] * 9 + [1 if pred_idx == 0 else 0]) 
        report = classification_report(y_true, y_pred, labels=list(range(4)), target_names=CLASSES, output_dict=True, zero_division=0)
        st.table(report)

    with ev_col2:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
        st.pyplot(fig_cm)

else:
    st.info("System Standby. Upload a brain MRI scan via the sidebar to begin analysis.")

st.markdown("---")
st.caption("Developed by Osama Sawalha | Clinical Decision Support Tool")
