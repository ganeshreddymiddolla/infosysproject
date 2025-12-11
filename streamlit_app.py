import streamlit as st
from PIL import Image
import numpy as np
import time

# ==========================================
# 1. CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(
    page_title="DogLens AI",
    page_icon="🐶",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM CSS (The UI "Skin")
# ==========================================
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide standard Streamlit elements for a cleaner app look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Modern Card Container */
    .stApp {
        background-color: #0e1117;
    }
    
    .glass-card {
        background: rgba(38, 39, 48, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 20px;
        box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.2);
    }

    /* Headings */
    h1 {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF914D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 10px;
    }
    
    h3 {
        color: #e0e0e0;
        font-weight: 600;
    }

    /* Custom Buttons (Metric-like display for options) */
    div[data-testid="stMetric"] {
        background-color: #1f2229;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 12px;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: scale(1.02);
        border-color: #FF4B4B;
    }

    /* Result Cards */
    .breed-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .confidence-badge {
        background-color: #00CC96;
        color: black;
        padding: 5px 12px;
        border-radius: 15px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    .fail-badge {
        background-color: #FF4B4B;
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        font-weight: 700;
    }

    /* Data Grid */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        margin-top: 20px;
    }
    .info-item {
        background: #1f2229;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #FF4B4B;
    }
    .info-label {
        color: #888;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .info-value {
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. YOUR MODEL LOGIC (Paste here)
# ==========================================
def predict_breed_logic(image):
    """
    ⚠️ CRITICAL: Replace the mock logic below with your actual model inference.
    Input: PIL Image
    Output: Dictionary with 'breed', 'confidence', 'details', and 'all_scores'
    """
    
    # ---------------------------------------------------------
    # --- PASTE YOUR MODEL CODE HERE (Preprocessing & Predict) --
    # ---------------------------------------------------------
    
    # MOCK DATA FOR UI DEMONSTRATION (Remove this when you paste your code)
    # Simulating a wait time for realism
    time.sleep(1.5) 
    
    # MOCK RESULT (Example: change to logic based on model output)
    # Let's pretend we found a Golden Retriever with 85% confidence
    
    mock_prediction_success = True  # Set to False to test the <50% UI
    
    if mock_prediction_success:
        return {
            "success": True,
            "breed": "Golden Retriever",
            "confidence": 88.5,  # Percentage
            "details": {
                "Origin": "Scotland",
                "Lifespan": "10-12 years",
                "Temperament": "Intelligent, Friendly, Devoted",
                "Group": "Sporting"
            },
            # Return all classes and their scores
            "all_scores": {
                "Golden Retriever": 0.885,
                "Labrador": 0.05,
                "German Shepherd": 0.02,
                "Beagle": 0.01,
                "Poodle": 0.01,
                "Bulldog": 0.005
            }
        }
    else:
        # Simulate Low Confidence
        return {
            "success": False,
            "confidence": 35.0,
            "message": "Confidence below threshold."
        }

# ==========================================
# 4. UI COMPONENTS
# ==========================================

def home_screen():
    st.markdown("<h1>🐾 DogLens AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; margin-bottom: 40px;'>Advanced Breed Recognition System</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown("### 📸 Live Camera")
        if st.button("Start Camera", key="btn_cam", use_container_width=True):
            st.session_state.mode = 'camera'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown("### 📂 Upload Photo")
        if st.button("Upload File", key="btn_upl", use_container_width=True):
            st.session_state.mode = 'upload'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def display_results(result, image):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_img, col_data = st.columns([1, 2])
    
    with col_img:
        st.image(image, use_container_width=True, caption="Analyzed Image")

    with col_data:
        if result['success'] and result['confidence'] >= 50:
            # --- SUCCESS UI ---
            st.markdown(f"""
                <span class="confidence-badge">Match: {result['confidence']:.1f}%</span>
                <div class="breed-title">{result['breed']}</div>
                <p style="color: #bbb;">We have identified this dog with high confidence.</p>
            """, unsafe_allow_html=True)
            
            # Details Grid
            details = result['details']
            st.markdown('<div class="info-grid">', unsafe_allow_html=True)
            for key, value in details.items():
                st.markdown(f"""
                    <div class="info-item">
                        <div class="info-label">{key}</div>
                        <div class="info-value">{value}</div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # --- FAILURE UI (< 50%) ---
            st.markdown(f"""
                <span class="fail-badge">Low Confidence: {result.get('confidence', 0):.1f}%</span>
                <div class="breed-title" style="color: #FF4B4B;">Unknown Breed</div>
                <p style="color: #bbb; margin-top: 15px;">
                    We could not identify a specific dog breed with sufficient accuracy (Threshold: 50%).
                </p>
                <div class="info-item" style="border-left: 4px solid #555; margin-top: 20px;">
                    <div class="info-label">Suggestion</div>
                    <div class="info-value">Try a clearer image or a different angle.</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- ALL CONFIDENCE SCORES SECTION ---
    if result['success'] and result['confidence'] >= 50:
        st.markdown("### 📊 Analysis Breakdown")
        with st.expander("View Confidence for All Classes", expanded=True):
            scores = result['all_scores']
            # Sort scores
            sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
            
            for breed, score in sorted_scores.items():
                col_name, col_bar, col_val = st.columns([2, 5, 1])
                with col_name:
                    st.write(f"**{breed}**")
                with col_bar:
                    st.progress(score)
                with col_val:
                    st.write(f"{score*100:.1f}%")

def main():
    # Initialize State
    if 'mode' not in st.session_state:
        st.session_state.mode = 'home'

    # --- NAVIGATION ---
    if st.session_state.mode == 'home':
        home_screen()

    elif st.session_state.mode == 'camera':
        if st.button("← Back to Home"):
            st.session_state.mode = 'home'
            st.rerun()
            
        st.markdown("### 📸 Scan via Camera")
        img_file = st.camera_input("Capture Image")
        if img_file is not None:
            image = Image.open(img_file)
            with st.spinner("Analyzing neural patterns..."):
                result = predict_breed_logic(image)
            display_results(result, image)

    elif st.session_state.mode == 'upload':
        if st.button("← Back to Home"):
            st.session_state.mode = 'home'
            st.rerun()

        st.markdown("### 📂 Select Image")
        img_file = st.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])
        if img_file is not None:
            image = Image.open(img_file)
            st.image(image, width=300)
            if st.button("Analyze Breed", type="primary"):
                with st.spinner("Analyzing neural patterns..."):
                    result = predict_breed_logic(image)
                display_results(result, image)

if __name__ == "__main__":
    main()
