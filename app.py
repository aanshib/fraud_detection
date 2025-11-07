# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import torch
import easyocr
import numpy as np
import io
import time
import plotly.graph_objects as go

# ============ CONFIG ============
MODEL_DIR = "./model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ PAGE SETUP ============
st.set_page_config(
    page_title="💳 Financial Fraud Detector",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============ STYLING ============
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #eef2ff 0%, #e0f2fe 100%);
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background: transparent;
}
h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif;
}
.card {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    height: 3rem;
    transition: 0.3s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #1d4ed8, #0891b2);
}
.predict-card {
    border-radius: 15px;
    text-align: center;
    padding: 1.5rem;
    font-weight: 600;
    color: white;
    margin-bottom: 1.5rem;
}
.fraud { background: linear-gradient(135deg, #ef4444, #dc2626); }
.safe { background: linear-gradient(135deg, #10b981, #059669); }
</style>
""", unsafe_allow_html=True)

# ============ LOADERS ============
@st.cache_resource(show_spinner=False)
def load_model(model_dir=MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_ocr():
    reader = easyocr.Reader(["en"], gpu=False)
    return reader

tokenizer, model = load_model()
ocr_reader = load_ocr()

# ============ HELPERS ============
def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
    label = "Fraud" if pred_idx == 1 else "Not Fraud"
    return label, probs

def extract_text_easyocr(img):
    result = ocr_reader.readtext(np.array(img))
    extracted = " ".join([r[1] for r in result])
    return extracted.strip()

def make_pie_chart(probs):
    labels = ["Not Fraud", "Fraud"]
    colors = ["#10b981", "#ef4444"]
    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=[round(probs[0]*100,2), round(probs[1]*100,2)],
            hole=0.4,
            marker=dict(colors=colors),
            textinfo="label+percent",
            pull=[0.05, 0],
        )]
    )
    fig.update_layout(
        showlegend=False,
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig

# ============ HEADER ============
st.markdown("<h1 style='text-align:center;color:#1e3a8a;'>💳 Financial Fraud SMS Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#475569;'>Detect fraudulent messages from text or SMS screenshots using AI.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ============ LAYOUT ============
col1, col2 = st.columns([0.6, 0.4], gap="large")

# --- LEFT: Input Section ---
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📥 Input Section")

    input_type = st.radio("Select Input Type:", ["✉️ Text Message", "🖼️ SMS Screenshot"], horizontal=True)

    if input_type == "✉️ Text Message":
        sms_text = st.text_area(
            "Enter or paste SMS text:",
            height=180,
            placeholder="e.g., Congratulations! You've won ₹50,000. Click here to claim..."
        )
        if st.button("🔍 Analyze SMS"):
            if sms_text.strip():
                with st.spinner("Analyzing message..."):
                    time.sleep(1)
                    label, probs = predict_text(sms_text)
                st.session_state["prediction"] = (label, probs, sms_text)
            else:
                st.warning("Please enter some text for prediction.")

    elif input_type == "🖼️ SMS Screenshot":
        uploaded_file = st.file_uploader("Upload SMS Image (JPG/PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
            if st.button("🧾 Extract & Analyze"):
                with st.spinner("Extracting text from image..."):
                    extracted_text = extract_text_easyocr(img)
                if extracted_text:
                    st.write("**Extracted Text:**")
                    st.code(extracted_text)
                    label, probs = predict_text(extracted_text)
                    st.session_state["prediction"] = (label, probs, extracted_text)
                else:
                    st.error("⚠️ No text detected. Try a clearer image.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- RIGHT: Result Section ---
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if "prediction" in st.session_state:
        label, probs, text = st.session_state["prediction"]

        # Prediction Card
        color_class = "fraud" if label == "Fraud" else "safe"
        st.markdown(
            f"<div class='predict-card {color_class}'><h2>{label}</h2>"
            f"<p>Confidence: {np.max(probs)*100:.2f}%</p></div>",
            unsafe_allow_html=True
        )

        # Pie Chart
        st.plotly_chart(make_pie_chart(probs), use_container_width=True)

        # Details
        st.write("### 💡 Confidence Breakdown")
        st.write(f"**Fraud Probability:** {probs[1]*100:.2f}%")
        st.write(f"**Not Fraud Probability:** {probs[0]*100:.2f}%")

    else:
        st.info("Upload a text or image to see prediction results.")
    st.markdown("</div>", unsafe_allow_html=True)

# ============ FOOTER ============
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#6b7280;'>🚀 Built with <b>Transformers</b> + <b>Streamlit</b> + <b>EasyOCR</b> + <b>Plotly</b><br>Designed for visual clarity and structured insight.</p>",
    unsafe_allow_html=True,
)
