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

from web_app.utils import preprocess_canvas_image, reshape_for_prediction
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
    st.markdown('### Instruction')
    st.markdown(
        "1. Draw a digit (0-9) inside the black box.\n"
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
    
    predict_btn = st.button("Predict 🚀", use_container_width=True, type="primary")

with col2:
    st.markdown('<div class="sub-header">Live Prediction Results:</div>', unsafe_allow_html=True)
    
    results_placeholder = st.empty()
    
    if predict_btn and canvas_result.image_data is not None:
        raw_image = canvas_result.image_data
        
        # Check if drawn (canvas gives non-zero pixels)
        if np.max(raw_image) > 0:
            # Preprocess
            normalized_img_2d = preprocess_canvas_image(raw_image)
            model_input = reshape_for_prediction(normalized_img_2d)
            
            # Predict
            selected_model = models_dict[model_choice]
            pred_probs = selected_model.predict(model_input)[0]
            pred_class = np.argmax(pred_probs)
            confidence = pred_probs[pred_class]
            
            with results_placeholder.container():
                st.success(f"**Predicted Digit:** {pred_class} (Confidence: {confidence*100:.2f}%)")
                
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
                    # Grad-CAM only works natively with Conv Layers, check if CNN string is in model choice
                    if 'Cnn' in model_choice or 'cnn' in model_choice.lower():
                        st.markdown('#### What the AI Sees (Grad-CAM)')
                        st.caption("Warm colors (red/orange) represent the regions that strongly activated the neural network's final layers.")
                        heatmap = grad_cam(selected_model, model_input[0])
                        # Apply to the pre-processed visual
                        overlayed_img = get_overlayed_image(normalized_img_2d, heatmap, alpha=0.5)
                        
                        # Resize for better view in web app
                        overlayed_img = cv2.resize(overlayed_img, (150, 150), interpolation=cv2.INTER_NEAREST)
                        
                        st.image(overlayed_img, width=150)
                    else:
                        st.info("Grad-CAM visualization is only supported for Convolutional Neural Networks (CNNs).")
                        
        else:
            results_placeholder.warning("Please draw a digit on the canvas first!")
    else:
        results_placeholder.info("Awaiting user input...")
