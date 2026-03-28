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
st.set_page_config(page_title="Digit Recognizer AI", page_icon="🔢", layout="wide")

# --- Header Stylings ---
st.markdown("""
<style>
    div[data-testid="stSidebar"] {
        background-color: #161618;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #007BFF, #00E1FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: left;
        padding-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #A0A0A0;
    }
    .canvas-container {
        display: flex;
        justify-content: center;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔢 Neural Digit Recognizer</div>', unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_all_models():
    models = {}
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_paths = glob.glob(os.path.join(base_dir, 'models', 'saved_models', '*.keras'))
    for path in model_paths:
        name = os.path.basename(path).replace('_best.keras', '').replace('_', ' ').title()
        model = tf.keras.models.load_model(path)
        models[name] = model
    return models

models_dict = load_all_models()

# --- Initialize Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# --- Sidebar Layout ---
with st.sidebar:
    st.markdown('### Model Configuration')
    if not models_dict:
        st.warning("No models found! Please train a model first.")
        st.stop()
        
    model_choice = st.selectbox(
        "Choose an architecture:",
        list(models_dict.keys())
    )
    
    st.markdown('### Explainable AI (XAI)')
    show_gradcam = st.toggle("Show Grad-CAM Heatmap", value=True, help="Visualize where the CNN is focusing to make its prediction.")
    
    st.markdown('---')
    st.markdown('### Prediction History')
    if not st.session_state.history:
        st.info("No predictions yet.")
    for item in reversed(st.session_state.history[-5:]): # Show last 5
        st.write(f"**{item['digit']}** ({item['model']}) - {item['confidence']:.1f}%")

    st.markdown('---')
    st.markdown('### Instruction')
    st.markdown(
        "1. Draw a digit OR Upload an image OR Select a sample.\n"
        "2. Click the **Predict** button.\n"
        "3. See the model's confidence across all classes."
    )

# --- Main Layout ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="sub-header">Draw Your Digit:</div>', unsafe_allow_html=True)
    
    # Custom CSS on top of st_canvas is tricky, so we wrap it
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=320,
        width=320,
        drawing_mode="freedraw",
        key="canvas",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    predict_btn = st.button("Predict Canvas 🚀", use_container_width=True, type="primary")

    st.markdown('---')
    st.markdown('<div class="sub-header">OR Upload Image:</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    st.markdown('---')
    st.markdown('<div class="sub-header">OR Try a Sample:</div>', unsafe_allow_html=True)
    sample_files = glob.glob('web_app/static/samples/*.png')
    if sample_files:
        sample_cols = st.columns(5)
        selected_sample = None
        for i, sample_path in enumerate(sample_files[:10]):
            with sample_cols[i % 5]:
                digit_label = os.path.basename(sample_path).split('_')[1].split('.')[0]
                if st.button(f"{digit_label}", key=f"btn_{i}"):
                    selected_sample = sample_path
    
    # Combined Predict Logic
    final_input_image = None
    source = None
    
    if predict_btn and canvas_result.image_data is not None:
        if np.max(canvas_result.image_data) > 0:
            final_input_image = preprocess_canvas_image(canvas_result.image_data)
            source = "Canvas"
            
    elif uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, 1)
        final_input_image = preprocess_uploaded_image(opencv_img)
        source = "Upload"
        
    elif selected_sample:
        opencv_img = cv2.imread(selected_sample)
        final_input_image = preprocess_uploaded_image(opencv_img)
        source = f"Sample ({os.path.basename(selected_sample)})"

with col2:
    results_placeholder = st.empty()
    
    if final_input_image is not None:
        model_input = reshape_for_prediction(final_input_image)
        
        # Predict
        selected_model = models_dict[model_choice]
        pred_probs = selected_model.predict(model_input)[0]
        pred_class = np.argmax(pred_probs)
        confidence = pred_probs[pred_class]
        
        # Save to history
        st.session_state.history.append({
            "digit": pred_class,
            "confidence": confidence * 100,
            "model": model_choice,
            "source": source
        })
        
        with results_placeholder.container():
            st.success(f"**Predicted Digit:** {pred_class} (Confidence: {confidence*100:.2f}%)")
            st.info(f"Source: {source}")
            
            # Confidence Chart
            fig = go.Figure(data=[
                go.Bar(
                    x=[str(i) for i in range(10)], 
                    y=pred_probs, 
                    marker_color=['#00E1FF' if i == pred_class else '#3A3A3C' for i in range(10)]
                )
            ])
            fig.update_layout(
                title="Model Confidence per Class",
                xaxis_title="Classification Digit",
                yaxis_title="Probability",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F2F2F7'),
                margin=dict(l=0, r=0, b=0, t=40)
            )
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            # XAI (Explainability)
            if show_gradcam:
                if 'Cnn' in model_choice or 'cnn' in model_choice.lower():
                    st.markdown('#### What the AI Sees (Grad-CAM)')
                    st.caption("Warm colors (red/orange) represent high activation regions.")
                    heatmap = grad_cam(selected_model, model_input[0])
                    overlayed_img = get_overlayed_image(final_input_image, heatmap, alpha=0.5)
                    overlayed_img = cv2.resize(overlayed_img, (150, 150), interpolation=cv2.INTER_NEAREST)
                    st.image(overlayed_img, width=150)
                else:
                    st.info("Grad-CAM visualization is only supported for CNNs.")
    else:
        results_placeholder.info("Awaiting user input (Draw, Upload, or select Sample)...")
