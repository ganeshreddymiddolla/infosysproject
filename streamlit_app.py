import streamlit as st
import numpy as np
import time
import os
from PIL import Image, ImageOps

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PawPrint ID",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. BRANDED UI DESIGN (CSS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    /* GLOBAL THEME */
    .stApp {
        background-color: #FFF8F0; /* Soft Warm White */
        font-family: 'Nunito', sans-serif;
        color: #2D3748;
    }
    
    /* HIDE DEFAULTS */
    header, footer, #MainMenu {visibility: hidden;}

    /* --- HOME SCREEN --- */
    .hero-section {
        text-align: center;
        padding: 50px 20px;
    }
    .brand-logo {
        font-size: 60px;
        margin-bottom: 10px;
        display: inline-block;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));
    }
    .brand-name {
        font-size: 48px;
        font-weight: 800;
        color: #D97706; /* Warm Amber */
        margin: 0;
        letter-spacing: -1px;
    }
    .brand-tagline {
        color: #718096;
        font-size: 18px;
        font-weight: 600;
    }

    /* UPLOAD BOX */
    .upload-card {
        background: white;
        border-radius: 30px;
        padding: 40px;
        box-shadow: 0 20px 40px rgba(217, 119, 6, 0.1);
        border: 2px dashed #FBD38D;
        text-align: center;
        transition: transform 0.2s;
        max-width: 700px;
        margin: 30px auto;
    }
    .upload-card:hover {
        border-color: #D97706;
        transform: scale(1.01);
    }

    /* --- RESULT DASHBOARD --- */
    .dashboard-container {
        padding: 20px;
    }
    
    /* ID CARD (Left) */
    .id-card {
        background: white;
        border-radius: 24px;
        padding: 0;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #EDF2F7;
    }
    .id-header {
        background: #FFFAF0;
        padding: 25px;
        border-bottom: 1px solid #FEEBC8;
        text-align: center;
    }
    .breed-name {
        font-size: 32px;
        font-weight: 800;
        color: #2D3748;
        margin: 10px 0 5px 0;
    }
    .match-badge {
        background: #C6F6D5;
        color: #22543D;
        padding: 6px 14px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        padding: 20px;
        gap: 20px;
    }
    .stat-box {
        background: #F7FAFC;
        padding: 15px;
        border-radius: 16px;
        text-align: center;
    }
    .stat-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #A0AEC0;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .stat-value {
        font-size: 16px;
        font-weight: 700;
        color: #4A5568;
        margin-top: 5px;
    }
    
    .bio-section {
        padding: 0 25px 25px 25px;
        color: #718096;
        font-size: 15px;
        line-height: 1.6;
        text-align: center;
    }

    /* CHATBOT (Right) */
    .chat-panel {
        background: white;
        border-radius: 24px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #EDF2F7;
        height: 700px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .chat-header-bar {
        background: #D97706;
        color: white;
        padding: 20px;
        font-weight: 700;
        font-size: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* MESSAGES */
    .chat-bubble {
        padding: 14px 18px;
        border-radius: 18px;
        max-width: 80%;
        font-size: 15px;
        line-height: 1.5;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .user-msg {
        background: #2D3748;
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 4px;
        margin-left: auto;
    }
    .bot-msg {
        background: #F7FAFC;
        color: #2D3748;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
        border: 1px solid #EDF2F7;
    }

    /* BUTTONS */
    .stButton > button {
        background-color: #D97706;
        color: white;
        border-radius: 15px;
        border: none;
        padding: 12px 0;
        font-weight: 700;
        font-size: 16px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #B7791F;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. EXPANDED KNOWLEDGE BASE
# ==========================================
BREED_DB = {
    "German Shepherd": {
        "Origin": "Germany", "Life": "7-10 years", "Group": "Herding", "Height": "22-26 in",
        "Bio": "A versatile working dog known for courage, loyalty, and high intelligence.",
        "Diet": "High-protein formula (22%+) to support strong muscles.",
        "Care": "Requires daily mental stimulation and brushing twice a week.",
        "Temperament": "Confident, Smart, Brave"
    },
    "Golden Retriever": {
        "Origin": "Scotland", "Life": "10-12 years", "Group": "Sporting", "Height": "21-24 in",
        "Bio": "Friendly and devoted. They are eager to please and make excellent family pets.",
        "Diet": "Balanced diet. Watch calories as they love to eat!",
        "Care": "Daily walks and regular grooming to manage shedding.",
        "Temperament": "Friendly, Intelligent, Devoted"
    },
    "Labrador Retriever": {
        "Origin": "Canada", "Life": "10-12 years", "Group": "Sporting", "Height": "21-24 in",
        "Bio": "Outgoing and active. The most popular breed in many countries.",
        "Diet": "Prone to obesity. Measure food carefully.",
        "Care": "High energy needs. Good for swimming and fetch.",
        "Temperament": "Outgoing, Gentle, Agile"
    },
    "Siberian Husky": {
        "Origin": "Siberia", "Life": "12-14 years", "Group": "Working", "Height": "20-23 in",
        "Bio": "Born to run. Known for their endurance and wolf-like appearance.",
        "Diet": "Rich in protein and fat, similar to working dogs.",
        "Care": "Heavy shedders (blow coat twice a year). Needs running.",
        "Temperament": "Loyal, Mischievous, Outgoing"
    },
    "Pug": {
        "Origin": "China", "Life": "13-15 years", "Group": "Toy", "Height": "10-13 in",
        "Bio": "A lot of dog in a small space. Charming and loving.",
        "Diet": "Low-calorie diet to prevent breathing issues from weight.",
        "Care": "Clean face wrinkles daily. Avoid extreme heat.",
        "Temperament": "Charming, Mischievous, Loving"
    },
    "Chihuahua": {
        "Origin": "Mexico", "Life": "14-16 years", "Group": "Toy", "Height": "5-8 in",
        "Bio": "A tiny dog with a huge personality. Very loyal to one person.",
        "Diet": "Nutrient-dense small kibble.",
        "Care": "Keep warm in cold weather. Dental care is crucial.",
        "Temperament": "Graceful, Sassy, Devoted"
    },
    "Rottweiler": {
        "Origin": "Germany", "Life": "9-10 years", "Group": "Working", "Height": "22-27 in",
        "Bio": "A robust and powerful guardian. Loyal to their family.",
        "Diet": "High quality protein for muscle mass.",
        "Care": "Early socialization is mandatory. Moderate grooming.",
        "Temperament": "Loyal, Confident, Fearless"
    },
    "Beagle": {
        "Origin": "UK", "Life": "10-15 years", "Group": "Hound", "Height": "13-15 in",
        "Bio": "Merry and curious. Guided by their powerful nose.",
        "Diet": "Strict portion control. They are food obsessed.",
        "Care": "Keep on leash; they will follow scents anywhere.",
        "Temperament": "Merry, Curious, Friendly"
    },
    "Bulldog": {
        "Origin": "UK", "Life": "8-10 years", "Group": "Non-Sporting", "Height": "14-15 in",
        "Bio": "Calm and courageous. Known for their loose skin and pushed-in nose.",
        "Diet": "Easy-to-digest food to reduce gas.",
        "Care": "Low energy, minimal exercise needed. Clean folds.",
        "Temperament": "Docile, Willful, Friendly"
    },
    "Poodle": {
        "Origin": "France/Germany", "Life": "10-18 years", "Group": "Non-Sporting", "Height": "Varies",
        "Bio": "Beneath the curly coat is an elegant athlete and genius mind.",
        "Diet": "High quality fat for coat health.",
        "Care": "Professional grooming required every 4-6 weeks.",
        "Temperament": "Active, Proud, Very Smart"
    }
}

FALLBACK_DATA = {
    "Origin": "International", "Life": "10-13 years", "Group": "Mixed/Unknown", "Height": "Varies",
    "Bio": "A loyal canine companion.",
    "Diet": "Standard balanced dog food appropriate for size.",
    "Care": "Regular vet checkups and daily walks.",
    "Temperament": "Loyal, Friendly"
}

# ==========================================
# 4. APP LOGIC
# ==========================================
MODEL_PATH = 'final_model.keras'
CLASSES_PATH = 'classes.txt'
CONFIDENCE_THRESHOLD = 50.0

if 'page' not in st.session_state: st.session_state.page = 'HOME'
if 'data' not in st.session_state: st.session_state.data = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

def reset():
    st.session_state.page = 'HOME'
    st.session_state.data = None
    st.session_state.chat_history = []
    st.rerun()

def get_info(name):
    # Fuzzy Search
    for key in BREED_DB:
        if key in name: return BREED_DB[key]
    return FALLBACK_DATA

def smart_bot_response(breed, text):
    text = text.lower()
    info = get_info(breed)
    
    if any(x in text for x in ["food", "diet", "eat", "feed", "hungry"]):
        return f"🍖 **Dietary Advice:** {info['Diet']}"
    elif any(x in text for x in ["groom", "brush", "bath", "care", "shed"]):
        return f"🛁 **Grooming & Care:** {info['Care']}"
    elif any(x in text for x in ["size", "height", "weight", "big", "small"]):
        return f"📏 **Size:** This breed typically stands {info['Height']} tall."
    elif any(x in text for x in ["origin", "from", "history"]):
        return f"🌍 **Origin:** The {breed} comes from {info['Origin']}."
    elif any(x in text for x in ["life", "age", "old", "live"]):
        return f"⏳ **Lifespan:** They typically live for {info['Life']}."
    else:
        return f"The **{breed}** is a {info['Group']} dog known for being {info['Temperament'].lower()}. Ask me about their diet, grooming, or lifespan!"

# ==========================================
# 5. UI: HOME PAGE
# ==========================================
if st.session_state.page == 'HOME':
    
    st.markdown("""
        <div class="hero-section">
            <div class="brand-logo">🐾</div>
            <h1 class="brand-name">PawPrint ID</h1>
            <p class="brand-tagline">Identify breeds & Chat with AI Experts</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("### 📸 Upload a Photo")
    st.markdown("<p style='color:#A0AEC0; margin-bottom:20px;'>Supports JPG, PNG</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("🔍 Analyzing breed characteristics..."):
            import tensorflow as tf
            if not os.path.exists(MODEL_PATH):
                st.error("⚠️ Model not found! Please run training script.")
                st.stop()
            
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASSES_PATH, 'r') as f: classes = [x.strip() for x in f.readlines()]
            
            image = Image.open(uploaded_file).convert('RGB')
            img_resized = ImageOps.fit(image, (224, 224), Image.LANCZOS)
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)
            score = preds[0]
            top_idx = np.argmax(score)
            conf = 100 * np.max(score)
            
            raw_name = classes[top_idx]
            # Name Cleaning
            breed_name = raw_name.split('-', 1)[1].replace('_', ' ').title() if '-' in raw_name else raw_name
            
            st.session_state.data = {"image": image, "breed": breed_name, "conf": conf}
            st.session_state.chat_history = [{"role": "assistant", "content": f"Woof! I see a **{breed_name}**. How can I help you care for them?"}]
            st.session_state.page = 'RESULT'
            st.rerun()

# ==========================================
# 6. UI: RESULT DASHBOARD
# ==========================================
elif st.session_state.page == 'RESULT':
    data = st.session_state.data
    breed = data['breed']
    conf = data['conf']
    
    # --- LOW CONFIDENCE CHECK ---
    if conf < CONFIDENCE_THRESHOLD:
        st.markdown(f"""
        <div style="text-align:center; padding:50px;">
            <div style="font-size:60px;">🐕❓</div>
            <h1 style="color:#C53030;">No Dog Detected</h1>
            <p style="color:#718096; font-size:18px;">
                Confidence: {conf:.1f}%<br>
                Our AI isn't sure this is a dog. Please try a clearer photo.
            </p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button("Try Again"): reset()
            
    # --- SUCCESS DASHBOARD ---
    else:
        info = get_info(breed)
        
        # Split Layout
        col_left, col_right = st.columns([1, 1.4], gap="large")
        
        # LEFT: ID CARD
        with col_left:
            st.markdown('<div class="id-card">', unsafe_allow_html=True)
            st.image(data['image'], use_container_width=True)
            
            st.markdown(f"""
                <div class="id-header">
                    <div class="match-badge">Match: {conf:.1f}%</div>
                    <h1 class="breed-name">{breed}</h1>
                    <p style="color:#718096; margin:0;">{info['Group']} Group</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-label">Origin</div>
                        <div class="stat-value">{info['Origin']}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Lifespan</div>
                        <div class="stat-value">{info['Life']}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Height</div>
                        <div class="stat-value">{info['Height']}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Temperament</div>
                        <div class="stat-value" style="font-size:14px;">{info['Temperament']}</div>
                    </div>
                </div>
                
                <div class="bio-section">
                    "{info['Bio']}"
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⬅ Scan New Dog"): reset()

        # RIGHT: CHATBOT
        with col_right:
            st.markdown(f"""
            <div class="chat-panel">
                <div class="chat-header-bar">
                    <span>💬</span> Ask about {breed}s
                </div>
            """, unsafe_allow_html=True)
            
            # Chat History Scroller
            chat_container = st.container(height=550)
            with chat_container:
                for msg in st.session_state.chat_history:
                    css = "user-msg" if msg["role"] == "user" else "bot-msg"
                    align = "flex-end" if msg["role"] == "user" else "flex-start"
                    
                    st.markdown(f"""
                        <div style="display:flex; justify-content:{align}; margin-bottom:10px;">
                            <div class="chat-bubble {css}">{msg["content"]}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True) # End chat panel

            # Chat Input (Outside the custom HTML box to allow interaction)
            if prompt := st.chat_input(f"Ask about diet, grooming, or training..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                reply = smart_bot_response(breed, prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()
