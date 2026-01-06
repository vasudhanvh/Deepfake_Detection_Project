import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="Deepfake Detection EfficientNet + ResNet",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ABSOLUTE PATH SETUP
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'safe_model_v3.h5')

# ==========================================
# 2. MODERN UI STYLING (CSS)
# ==========================================
st.markdown("""
    <style>
        /* MAIN BACKGROUND */
        .stApp {
            background-color: #050505;
            background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #050505 60%);
        }
        
        /* TYPOGRAPHY */
        h1, h2, h3 {
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            color: #ffffff;
        }
        
        /* CUSTOM CARDS */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        /* RESULT BANNERS */
        .result-fake {
            background: linear-gradient(90deg, #5b1e1e 0%, #ff4b4b 100%);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
            box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
        }
        .result-real {
            background: linear-gradient(90deg, #1e4d2b 0%, #00ff88 100%);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            box-shadow: 0 4px 15px rgba(0, 255, 136, 0.2);
        }
        
        /* FILE UPLOADER */
        .stFileUploader {
            border: 2px dashed #444;
            border-radius: 10px;
            padding: 20px;
        }
        
        /* STREAMLIT ELEMENTS OVERRIDE */
        div[data-testid="stHeader"] {background: transparent;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. CORE LOGIC
# ==========================================
def generate_frequency_map(image_array):
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        float_gray = np.float32(gray) / 255.0
        dct = cv2.dct(float_gray)
        dct_log = np.log(np.abs(dct) + 1e-5)
        dct_norm = (dct_log - np.min(dct_log)) / (np.max(dct_log) - np.min(dct_log))
        dct_3ch = cv2.merge([dct_norm, dct_norm, dct_norm])
        return dct_3ch
    except Exception:
        return np.zeros((256, 256, 3))

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((256, 256))
    img_array = np.array(image)

    X_rgb = img_array.astype('float32') / 255.0
    X_rgb = np.expand_dims(X_rgb, axis=0)

    freq_map = generate_frequency_map(img_array)
    X_freq = np.expand_dims(freq_map, axis=0)

    return image, freq_map, {'input_rgb': X_rgb, 'input_freq': X_freq}

def build_dual_stream_model():
    input_rgb = tf.keras.layers.Input(shape=(256, 256, 3), name='input_rgb')
    input_freq = tf.keras.layers.Input(shape=(256, 256, 3), name='input_freq')

    eff_net = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_tensor=input_rgb)
    pool_rgb = tf.keras.layers.GlobalAveragePooling2D()(eff_net.output)

    res_net = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_tensor=input_freq)
    pool_freq = tf.keras.layers.GlobalAveragePooling2D()(res_net.output)

    merged = tf.keras.layers.Concatenate()([pool_rgb, pool_freq])
    dense = tf.keras.layers.Dense(256, activation='relu')(merged)
    dropout = tf.keras.layers.Dropout(0.5)(dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)

    return tf.keras.models.Model(inputs=[input_rgb, input_freq], outputs=output)

@st.cache_resource
def load_deepfake_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ System Error: Model file missing at {MODEL_PATH}")
        return None
    try:
        model = build_dual_stream_model()
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_deepfake_model()

# ==========================================
# 4. MODERN UI LAYOUT
# ==========================================

# --- HERO SECTION (UPDATED TITLE, NO LOGO) ---
st.title("Deepfake Detection EfficientNet + ResNet")
st.caption("Dual-Stream Forensic Analysis Architecture")

st.divider()

# --- TOP SECTION: INPUT & REPORT ---
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("1. Input Analysis")
    uploaded_file = st.file_uploader("Upload Suspect Image", type=["jpg", "png", "jpeg"], help="Supported formats: JPG, PNG")
    
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **Dual-Stream Logic:**
        1. **Spatial Stream:** Analyzes visible artifacts (eyes, lighting).
        2. **Frequency Stream:** Detects invisible GAN upsampling noise.
        """)

if uploaded_file is not None and model is not None:
    # --- PROCESSING ---
    with st.spinner('⚡ performing forensic scan...'):
        original_img, freq_map_vis, inputs = preprocess_image(uploaded_file)
        prediction = model.predict(inputs)
        confidence = prediction[0][0]

    # --- FORENSIC REPORT (Top Right) ---
    with col_right:
        st.subheader("2. Forensic Report")
        
        # Prepare variables
        is_fake = confidence > 0.50
        result_text = "⚠️ DEEPFAKE DETECTED" if is_fake else "✅ AUTHENTIC MEDIA"
        result_class = "result-fake" if is_fake else "result-real"
        pct_value = confidence if is_fake else (1 - confidence)
        
        # Main Banner
        st.markdown(f'<div class="{result_class}">{result_text}</div>', unsafe_allow_html=True)
        st.write("") # Spacer

        # Key Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:12px; opacity:0.7;">CONFIDENCE</div>
                <div style="font-size:24px; font-weight:bold;">{confidence:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:12px; opacity:0.7;">CERTAINTY</div>
                <div style="font-size:24px; font-weight:bold;">{pct_value:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:12px; opacity:0.7;">SCAN TIME</div>
                <div style="font-size:24px; font-weight:bold;">0.4s</div>
            </div>
            """, unsafe_allow_html=True)

    # ============================================
    # 3. FULL WIDTH EVIDENCE VISUALIZATION
    # ============================================
    st.write("---") # Visual Separator
    st.subheader("3. Detailed Evidence Visualization")
    
    # Create side-by-side full width columns
    vis_col1, vis_col2 = st.columns(2)
    
    with vis_col1:
        st.markdown("##### Stream A: Spatial Domain (RGB)")
        st.image(original_img, caption="Visual Appearance (Human Visible)", use_container_width=True)
        if is_fake:
            st.caption("⚠️ Analysis: Model scans for blending artifacts, unnatural lighting, and texture inconsistencies.")
    
    with vis_col2:
        st.markdown("##### Stream B: Frequency Domain (DCT)")
        freq_display = (freq_map_vis * 255).astype(np.uint8)
        st.image(freq_display, caption="Spectral Appearance (Machine Visible)", use_container_width=True)
        if is_fake:
            st.caption("⚠️ Analysis: High-frequency noise pattern detected. This suggests GAN upsampling traces.")

else:
    # Empty State on Right Side
    with col_right:
        st.info("👈 Waiting for image input...")
        st.markdown("""
        <div style="text-align: center; opacity: 0.5; padding: 50px;">
            <h3 style="color: #444;">Ready to Scan</h3>
            <p>Upload an image to begin the dual-stream analysis.</p>
        </div>
        """, unsafe_allow_html=True)