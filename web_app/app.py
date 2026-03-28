import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
import glob
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import cv2

# Adjust path to find src and utils
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from web_app.utils import preprocess_canvas_image, reshape_for_prediction, preprocess_uploaded_image
from src.explainability import grad_cam, get_overlayed_image

# --- Page Config ---
st.set_page_config(page_title="DigitMind AI", page_icon="🧠", layout="wide")

# --- Header Stylings ---
st.markdown("""
<style>
    div[data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #30363D;
    }
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #FF3CAC, #784BA0, #2B86C5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: left;
        padding-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #F0F6FC;
        margin-bottom: 1rem;
    }
    .canvas-container {
        display: flex;
        justify-content: center;
        border: 2px solid #30363D;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .sample-container {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        transition: transform 0.2s;
    }
    .sample-container:hover {
        transform: translateY(-5px);
        border-color: #58A6FF;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 DigitMind AI</div>', unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_all_models():
    models = {}
    # Try multiple base directories for local & Streamlit Cloud compatibility
    candidates = [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # parent of web_app/
        os.getcwd(),                                                    # current working dir
        os.path.dirname(os.path.abspath(__file__)),                    # web_app/ itself
    ]
    model_paths = []
    for base_dir in candidates:
        model_paths = glob.glob(os.path.join(base_dir, 'models', 'saved_models', '*.keras'))
        if model_paths:
            break
    for path in model_paths:
        name = os.path.basename(path).replace('_best.keras', '').replace('_', ' ').title()
        try:
            model = tf.keras.models.load_model(path)
            models[name] = model
        except Exception as e:
            st.warning(f"Could not load model {os.path.basename(path)}: {e}")
    return models

models_dict = load_all_models()

# --- Initialize Session State ---
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = None

# --- Sidebar Layout ---
with st.sidebar:
    st.markdown('### ⚙️ Configuration')
    if not models_dict:
        st.error("No pre-trained models found. Please check that `.keras` files exist in `models/saved_models/`.")
        st.stop()
        
    model_choice = st.selectbox(
        "Model Architecture:",
        list(models_dict.keys()),
        help="Select which pre-trained neural network to use for prediction."
    )
    
    st.markdown('### 🛠️ Explainable AI (XAI)')
    show_gradcam = st.toggle("Enable Grad-CAM", value=True, help="Heatmap visualization showing where the model focused.")
    
    st.markdown('---')
    st.markdown('### 📖 Instructions')
    st.markdown(
        "1. **Draw** a digit on the canvas\n"
        "2. **Upload** a custom image\n"
        "3. **Select** a sample from the gallery\n\n"
        "The model will provide the most likely digit and a visual focus map."
    )
    
    st.markdown('---')
    st.markdown('**Version:** 1.2.0')

# --- Main Layout ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="sub-header">🎨 Digit Canvas:</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=22,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=320,
        width=320,
        drawing_mode="freedraw",
        key="canvas",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    predict_btn = st.button("Predict Canvas ✨", use_container_width=True, type="primary")

    st.markdown('---')
    st.markdown('<div class="sub-header">📂 Upload Image:</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    st.markdown('---')
    st.markdown('<div class="sub-header">🖼️ Sample Gallery:</div>', unsafe_allow_html=True)
    sample_files = sorted(glob.glob('web_app/static/samples/*.png'))
    
    if sample_files:
        sample_cols = st.columns(5)
        for i, sample_path in enumerate(sample_files[:10]):
            with sample_cols[i % 5]:
                # Display the image
                st.image(sample_path, use_container_width=True)
                # Select button below
                digit_label = os.path.basename(sample_path).split('_')[1].split('.')[0]
                if st.button(f"#{digit_label}", key=f"btn_{i}", use_container_width=True):
                    st.session_state.selected_sample = sample_path
    
    # Combined Predict Logic
    final_input_image = None
    source = None
    
    # Handle Canvas Input
    if predict_btn and canvas_result.image_data is not None:
        if np.max(canvas_result.image_data) > 0:
            final_input_image = preprocess_canvas_image(canvas_result.image_data)
            source = "Digital Canvas"
            st.session_state.selected_sample = None # Reset sample selection
            
    # Handle Upload Input
    elif uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, 1)
        final_input_image = preprocess_uploaded_image(opencv_img)
        source = "User Upload"
        st.session_state.selected_sample = None # Reset sample selection
        
    # Handle Sample Selection
    elif st.session_state.selected_sample:
        opencv_img = cv2.imread(st.session_state.selected_sample)
        final_input_image = preprocess_uploaded_image(opencv_img)
        source = f"Sample Digit ({os.path.basename(st.session_state.selected_sample).split('_')[1].split('.')[0]})"

with col2:
    results_placeholder = st.empty()
    
    if final_input_image is not None:
        model_input = reshape_for_prediction(final_input_image)
        
        # Predict
        selected_model = models_dict[model_choice]
        pred_probs = selected_model.predict(model_input, verbose=0)[0]
        pred_class = np.argmax(pred_probs)
        confidence = pred_probs[pred_class]
        
        with results_placeholder.container():
            st.markdown(f"### 🎯 Result: Digit **{pred_class}**")
            st.progress(float(confidence), text=f"Confidence: {confidence*100:.2f}%")
            st.caption(f"Processed via: {source}")
            
            # Confidence Chart
            fig = go.Figure(data=[
                go.Bar(
                    x=[str(i) for i in range(10)], 
                    y=pred_probs, 
                    marker=dict(
                        color=['#FF3CAC' if i == pred_class else '#3A3A3C' for i in range(10)],
                        line=dict(color='#F0F6FC', width=1)
                    )
                )
            ])
            fig.update_layout(
                title="Probability Distribution",
                xaxis_title="Predicted Digit",
                yaxis_title="Probability Score",
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F2F2F7'),
                margin=dict(l=0, r=0, b=0, t=50)
            )
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            # XAI (Explainability)
            if show_gradcam:
                if 'Cnn' in model_choice or 'cnn' in model_choice.lower():
                    st.markdown('---')
                    st.markdown('#### 🔍 Visual Focus Map (Grad-CAM)')
                    st.caption("Warm regions indicate where the model detected digit features.")
                    
                    try:
                        heatmap = grad_cam(selected_model, model_input[0])
                        overlayed_img = get_overlayed_image(final_input_image, heatmap, alpha=0.5)
                        overlayed_img = cv2.resize(overlayed_img, (180, 180), interpolation=cv2.INTER_NEAREST)
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(final_input_image, caption="Original Input", width=150)
                        with c2:
                            st.image(overlayed_img, caption="AI Focus Area", width=150)
                    except Exception as e:
                        st.error("⚠️ Visual Focus Map (Grad-CAM) could not be generated for this prediction.")
                        st.caption(f"Reason: {type(e).__name__}. The model might require re-training to support XAI features.")
                else:
                    st.info("💡 Grad-CAM visualization is only supported for CNN architectures.")
    else:
        results_placeholder.info("👋 Awaiting input. Please draw on the canvas, upload a file, or select a sample digit.")
