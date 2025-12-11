import streamlit as st
import numpy as np
import time
import os
from PIL import Image, ImageOps

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PawPedia AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CSS STYLING (Modern & Clean)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* GLOBAL RESET */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Poppins', sans-serif;
        color: #1E293B;
    }
    
    /* HIDE DEFAULT STREAMLIT ELEMENTS */
    header, footer, #MainMenu {visibility: hidden;}

    /* --- GIANT UPLOAD AREA (HOME PAGE) --- */
    .big-upload-box {
        background: white;
        padding: 60px;
        border-radius: 30px;
        border: 2px dashed #CBD5E1;
        box-shadow: 0 20px 50px rgba(0,0,0,0.05);
        text-align: center;
        max-width: 800px;
        margin: 40px auto;
        transition: all 0.3s ease;
    }
    .big-upload-box:hover {
        border-color: #6366F1;
        transform: translateY(-5px);
    }
    .icon-large {
        font-size: 80px;
        margin-bottom: 20px;
    }
    .title-large {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    /* MAKE FILE UPLOADER HUGE */
    [data-testid='stFileUploader'] {
        width: 100%;
        padding: 20px;
    }
    [data-testid='stFileUploader'] section {
        padding: 40px;
        background-color: #F1F5F9;
        border-radius: 20px;
    }
    [data-testid='stFileUploader'] button {
        display: none; /* Hide the small button inside */
    }

    /* --- PROFILE CARD (LEFT SIDE) --- */
    .profile-card {
        background: white;
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        height: 100%;
    }
    .breed-title {
        font-size: 32px;
        font-weight: 700;
        color: #1E293B;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .confidence-tag {
        background: #EEF2FF;
        color: #4F46E5;
        padding: 8px 16px;
        border-radius: 100px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    
    /* DATA TABLE */
    .data-row {
        display: flex;
        justify-content: space-between;
        padding: 15px 0;
        border-bottom: 1px solid #F1F5F9;
    }
    .data-label { color: #64748B; font-weight: 500; }
    .data-value { color: #0F172A; font-weight: 600; text-align: right; }

    /* --- CHATBOT (RIGHT SIDE) --- */
    .chat-box {
        background: white;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        padding: 20px;
        height: 650px;
        display: flex;
        flex-direction: column;
    }
    .chat-header {
        font-size: 18px;
        font-weight: 700;
        color: #334155;
        padding-bottom: 15px;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* MESSAGES */
    .msg-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        overflow-y: auto;
        height: 100%;
        padding-right: 10px;
    }
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 18px;
        max-width: 85%;
        font-size: 14px;
        line-height: 1.5;
        position: relative;
    }
    .user-bubble {
        background: #4F46E5;
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 4px;
    }
    .bot-bubble {
        background: #F1F5F9;
        color: #1E293B;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
    }

    /* BUTTONS */
    .stButton > button {
        background-color: #4F46E5;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 0;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #4338CA;
        transform: translateY(-2px);
    }
    
    /* ERROR UI */
    .error-box {
        background: #FEF2F2;
        border: 2px solid #FECACA;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        max-width: 500px;
        margin: 50px auto;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. KNOWLEDGE BASE (Smart Data)
# ==========================================
BREED_DATA = {
    "Golden Retriever": {
        "Origin": "Scotland", "Life": "10-12 yrs", "Group": "Sporting",
        "Bio": "Friendly, intelligent, and devoted. They are eager to please and love water.",
        "Diet": "2-3 cups of high-quality dry food. Avoid grains if allergic.",
        "Training": "Very easy to train. Use treats and praise.",
        "Health": "Watch for hip dysplasia and ear infections."
    },
    "German Shepherd": {
        "Origin": "Germany", "Life": "7-10 yrs", "Group": "Herding",
        "Bio": "Confident, courageous, and smart. A loyal guardian and working dog.",
        "Diet": "High-protein diet for muscle maintenance.",
        "Training": "Requires firm, consistent leadership.",
        "Health": "Prone to joint issues. Keep active."
    },
    "Labrador Retriever": {
        "Origin": "Canada", "Life": "10-12 yrs", "Group": "Sporting",
        "Bio": "Friendly, active, and outgoing. Great family pets.",
        "Diet": "Prone to obesity. Measure food carefully.",
        "Training": "Responds well to positive reinforcement.",
        "Health": "Joint health is a priority."
    },
    "Pug": {
        "Origin": "China", "Life": "13-15 yrs", "Group": "Toy",
        "Bio": "Charming, mischievous, and loving.",
        "Diet": "Low-calorie diet to prevent weight gain.",
        "Training": "Can be stubborn. Be patient.",
        "Health": "Keep cool in hot weather (Brachycephalic)."
    },
    "Siberian Husky": {
        "Origin": "Siberia", "Life": "12-14 yrs", "Group": "Working",
        "Bio": "Loyal, outgoing, and mischievous. Born to run.",
        "Diet": "Fish-based diet rich in Omega-3.",
        "Training": "Independent thinkers. Keep it fun.",
        "Health": "Check eyes regularly."
    },
    # Fallback
    "default": {
        "Origin": "Unknown", "Life": "10-13 yrs", "Group": "Companion",
        "Bio": "A loyal dog breed identified by AI.",
        "Diet": "Balanced commercial dog food.",
        "Training": "Positive reinforcement is key.",
        "Health": "Annual vet checkups recommended."
    }
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

def reset_app():
    st.session_state.page = 'HOME'
    st.session_state.data = None
    st.session_state.chat_history = []
    st.rerun()

def get_breed_info(name):
    for key in BREED_DATA:
        if key in name: return BREED_DATA[key]
    return BREED_DATA["default"]

def smart_reply(breed, text):
    text = text.lower()
    info = get_breed_info(breed)
    
    if any(x in text for x in ["food", "diet", "eat", "feed"]):
        return f"🍖 **Diet:** {info['Diet']}"
    elif any(x in text for x in ["train", "teach", "sit"]):
        return f"🎓 **Training:** {info['Training']}"
    elif any(x in text for x in ["health", "sick", "care"]):
        return f"❤️ **Health:** {info['Health']}"
    elif any(x in text for x in ["origin", "from", "history"]):
        return f"🌍 **Origin:** {info['Origin']}."
    else:
        return f"The **{breed}** is known for being {info['Bio'].lower()} Ask me about their diet or training!"

# ==========================================
# 5. UI: HOME PAGE (GIANT UPLOAD)
# ==========================================
if st.session_state.page == 'HOME':
    
    st.markdown("""
    <div class="big-upload-box">
        <div class="icon-large">🐶</div>
        <h1 class="title-large">PawPedia AI</h1>
        <p style="color:#64748B; font-size: 18px;">
            The Ultimate Dog Identifier & Chatbot
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Giant Upload Area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Custom Label
        st.markdown("<h3 style='text-align:center; color:#475569;'>Upload a photo below</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'], label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("🧠 AI is analyzing breed features..."):
            import tensorflow as tf
            
            if not os.path.exists(MODEL_PATH):
                st.error("Model missing.")
                st.stop()
                
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASSES_PATH, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            
            # Predict
            image = Image.open(uploaded_file).convert('RGB')
            img_resized = ImageOps.fit(image, (224, 224), Image.LANCZOS)
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)
            score = preds[0]
            top_idx = np.argmax(score)
            conf = 100 * np.max(score)
            
            raw_name = classes[top_idx]
            breed_name = raw_name.split('-', 1)[1].replace('_', ' ').title() if '-' in raw_name else raw_name
            
            st.session_state.data = {"image": image, "breed": breed_name, "conf": conf}
            st.session_state.chat_history = [{"role": "bot", "content": f"Hello! I see this is a **{breed_name}**. Ask me anything about them!"}]
            
            st.session_state.page = 'RESULT'
            st.rerun()

# ==========================================
# 6. UI: RESULT PAGE (SPLIT VIEW)
# ==========================================
elif st.session_state.page == 'RESULT':
    data = st.session_state.data
    breed = data['breed']
    conf = data['conf']
    
    # --- ERROR STATE ---
    if conf < CONFIDENCE_THRESHOLD:
        st.markdown(f"""
        <div class="error-box">
            <div style="font-size:50px;">🐕❓</div>
            <h2 style="color:#991B1B;">No Dog Detected</h2>
            <p style="color:#7F1D1D;">Confidence: {conf:.1f}%</p>
            <p style="color:#7F1D1D; margin-top:10px;">
                Our AI isn't sure this is a dog. Please upload a clearer photo.
            </p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button("Try Again"): reset_app()
            
    # --- SUCCESS STATE ---
    else:
        info = get_breed_info(breed)
        
        # Split Screen Layout
        left_col, right_col = st.columns([1, 1.3], gap="large")
        
        # LEFT: Profile Card
        with left_col:
            st.image(data['image'], use_container_width=True)
            
            # Clean HTML Block (No Breaks)
            st.markdown(f"""
            <div class="profile-card">
                <span class="confidence-tag">Match: {conf:.1f}%</span>
                <h1 class="breed-title">{breed}</h1>
                <p style="color:#64748B; margin-bottom:20px;">{info['Bio']}</p>
                
                <div class="data-row">
                    <span class="data-label">Origin</span>
                    <span class="data-value">{info['Origin']}</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Lifespan</span>
                    <span class="data-value">{info['Life']}</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Group</span>
                    <span class="data-value">{info['Group']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⬅ Upload New Photo"): reset_app()

        # RIGHT: Chat Interface
        with right_col:
            st.markdown(f"""
            <div class="chat-header">
                <span>💬</span> Chat with {breed} Expert
            </div>
            """, unsafe_allow_html=True)
            
            # Message History
            chat_container = st.container(height=500)
            with chat_container:
                for msg in st.session_state.chat_history:
                    bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
                    align = "flex-end" if msg["role"] == "user" else "flex-start"
                    
                    st.markdown(f"""
                    <div style="display:flex; justify-content:{align}; margin-bottom:10px;">
                        <div class="chat-bubble {bubble_class}">
                            {msg["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Input
            if prompt := st.chat_input(f"Ask about {breed}..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                reply = smart_reply(breed, prompt)
                st.session_state.chat_history.append({"role": "bot", "content": reply})
                st.rerun()
