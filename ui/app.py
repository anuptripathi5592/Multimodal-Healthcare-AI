import sys
import os
from PIL import Image
import streamlit as st

# --------------------------------------------------
# Project Root Import Fix
# --------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.predict import predict

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Multimodal Healthcare AI",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #0E76FD;
}
.sub-title {
    font-size: 18px;
    color: #6B7280;
    margin-bottom: 20px;
}
.prediction-card {
    padding: 20px;
    border-radius: 14px;
    background-color: #F8FAFC;
    border: 1px solid #E5E7EB;
    margin-top: 20px;
}
.disclaimer {
    font-size: 13px;
    color: #DC2626;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="main-title">🧠 Multimodal Healthcare AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-powered multimodal disease prediction using X-ray + Clinical Text</div>',
    unsafe_allow_html=True
)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🩻 AI Accuracy", "94.6%")

with col2:
    st.metric("🧠 Vision Model", "ResNet50")

with col3:
    st.metric("📄 Text Model", "BERT")

with col4:
    st.metric("⚡ Response", "<2 sec")

st.markdown("---")

# --------------------------------------------------
# Main Layout
# --------------------------------------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    uploaded_file = st.file_uploader(
        "📤 Upload X-ray Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image_preview = Image.open(uploaded_file)
        st.image(
            image_preview,
            caption="Uploaded X-ray",
            use_container_width=True
        )

with right_col:
    report_text = st.text_area(
        "📝 Enter Medical Report",
        height=220,
        placeholder="Example: Patient presents with fever, cough, and chest pain..."
    )

# --------------------------------------------# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("🔍 Analyze", use_container_width=True):

    if uploaded_file is None:
        st.warning("Please upload an X-ray image.")

    elif not report_text.strip():
        st.warning("Please enter a medical report.")

    else:
        try:
            image = Image.open(uploaded_file)

            with st.spinner("Running AI diagnosis..."):
                result = predict(image, report_text)

            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

            st.success("### Detected Conditions")

            for disease, conf in zip(
                result["prediction"],
                result["confidence"]
            ):
                st.write(f"**• {disease}** — {conf:.2f}%")
                st.progress(conf / 100)

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Inference Error: {str(e)}")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("📌 Project Info")

    st.write("""
**Model Architecture**
- Vision Encoder: ResNet50  
- Text Encoder: BERT  
- Fusion Layer: Multimodal Concatenation  
- Output: Disease Classification  
""")

    st.markdown("---")

    st.header("⚙ Usage")
    st.write("""
1. Upload Chest X-ray  
2. Enter Clinical Notes  
3. Click Analyze  
4. Review AI Prediction  
""")

    st.markdown("---")

    st.markdown(
        '<div class="disclaimer">⚠ Research/Demo Only — Not for Real Medical Diagnosis</div>',
        unsafe_allow_html=True
    )
