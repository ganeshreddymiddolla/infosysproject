import streamlit as st
import numpy as np
import time
import os
import random
from PIL import Image, ImageOps

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Lens AI",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. INSTANT-LOAD CSS (Lens + Chat Styles)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

    /* GLOBAL THEME */
    .stApp {
        background-color: #f0f2f5;
        font-family: 'Google Sans', sans-serif;
        color: #202124;
    }
    header, footer, #MainMenu {visibility: hidden;}

    /* LENS UI STYLES */
    .hero-container {
        text-align: center;
        padding: 40px 20px 20px;
        animation: fadeIn 0.8s ease-out;
    }
    .lens-logo {
        font-size: 50px;
        margin-bottom: 10px;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    .result-sheet {
        background: white;
        border-radius: 30px 30px 0 0;
        margin-top: -30px;
        padding: 30px;
        box-shadow: 0 -10px 40px rgba(0,0,0,0.1);
        position: relative;
        z-index: 100;
        animation: slideUp 0.5s ease-out;
    }
    
    /* CHAT UI STYLES */
    .chat-bubble {
        padding: 15px 20px;
        border-radius: 20px;
        margin-bottom: 10px;
        max-width: 80%;
        font-size: 15px;
        line-height: 1.5;
    }
    .user-msg {
        background-color: #0b57d0; /* Google Blue */
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    .bot-msg {
        background-color: white;
        color: #1f1f1f;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-bottom-left-radius: 5px;
    }
    
    /* ANIMATIONS */
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideUp { from { transform: translateY(100%); } to { transform: translateY(0); } }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        background: white;
        border-radius: 50px;
        padding: 5px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 40px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8f0fe;
        color: #1a73e8;
    }
    
    /* BUTTONS */
    .stButton > button {
        border-radius: 50px;
        background-color: #1a73e8;
        color: white;
        border: none;
        height: 50px;
        font-weight: 500;
        width: 100%;
    }
    .stButton > button:hover { background-color: #1557b0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. KNOWLEDGE BASE (Used for Lens & Chat)
# ==========================================
BREED_DB = {
    "Golden Retriever": {"Origin": "Scotland", "Life Span": "10-12 yrs", "Temperament": "Friendly, Intelligent, Devoted", "Diet": "High-quality dry food, 2-3 cups daily."},
    "Labrador Retriever": {"Origin": "Canada", "Life Span": "10-12 yrs", "Temperament": "Outgoing, Gentle, Agile", "Diet": "Protein-rich food, watch out for overeating."},
    "German Shepherd": {"Origin": "Germany", "Life Span": "7-10 yrs", "Temperament": "Confident, Smart, Brave", "Diet": "High-calorie diet for active working dogs."},
    "Bulldog": {"Origin": "UK", "Life Span": "8-10 yrs", "Temperament": "Docile, Willful, Friendly", "Diet": "Easily digestible food to prevent gas."},
    "Beagle": {"Origin": "UK", "Life Span": "10-15 yrs", "Temperament": "Merry, Curious, Friendly", "Diet": "Measured portions, they love to eat!"},
    "Poodle": {"Origin": "France", "Life Span": "10-18 yrs", "Temperament": "Active, Proud, Smart", "Diet": "Balanced diet rich in healthy fats for coat."},
    "Rottweiler": {"Origin": "Germany", "Life Span": "9-10 yrs", "Temperament": "Loyal, Loving, Confident", "Diet": "High protein content (22-26%) for muscle."},
    "Siberian Husky": {"Origin": "Siberia", "Life Span": "12-14 yrs", "Temperament": "Loyal, Mischievous", "Diet": "Fish-based diet rich in Omega-3."},
    "Pug": {"Origin": "China", "Life Span": "13-15 yrs", "Temperament": "Charming, Loving", "Diet": "Low-calorie food to prevent obesity."},
    "Chihuahua": {"Origin": "Mexico", "Life Span": "14-16 yrs", "Temperament": "Sassy, Graceful", "Diet": "Small kibble, nutrient-dense."},
}

FALLBACK_DB = {"Origin": "Unknown", "Life Span": "10-13 yrs", "Temperament": "Loyal Companion", "Diet": "Standard balanced dog food."}

# ==========================================
# 4. BACKEND LOGIC
# ==========================================
MODEL_PATH = 'final_model.keras'
CLASSES_PATH = 'classes.txt'
CONFIDENCE_THRESHOLD = 50.0

# Initialize Session State
if 'page' not in st.session_state: st.session_state.page = 'HOME'
if 'result' not in st.session_state: st.session_state.result = None
if 'messages' not in st.session_state: 
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Woof! I'm your Dog AI. You can upload a photo to identify a breed, or ask me questions here about training, diet, or breeds!"}
    ]

def reset():
    st.session_state.page = 'HOME'
    st.session_state.result = None
    st.rerun()

def get_details(name):
    for key in BREED_DB:
        if key in name: return BREED_DB[key]
    return FALLBACK_DB

def chat_response(prompt):
    """Simple rule-based logic to answer dog questions without an API key"""
    prompt = prompt.lower()
    
    # Check for breed specific questions
    found_breed = None
    for breed in BREED_DB:
        if breed.lower() in prompt:
            found_breed = breed
            break
            
    if found_breed:
        info = BREED_DB[found_breed]
        if "diet" in prompt or "food" in prompt:
            return f"🍖 **Diet for {found_breed}:** {info['Diet']}"
        elif "life" in prompt or "age" in prompt:
            return f"⏳ **Lifespan:** {found_breed}s typically live for {info['Life Span']}."
        elif "origin" in prompt or "from" in prompt:
            return f"🌍 **Origin:** The {found_breed} comes from {info['Origin']}."
        else:
            return f"🐶 **{found_breed}:** They are known to be {info['Temperament'].lower()}."

    # General Dog Questions
    if "hello" in prompt or "hi" in prompt:
        return "Hello! I can identify dog breeds from photos or answer questions about them. Try asking 'What do Pugs eat?'"
    elif "thank" in prompt:
        return "You're welcome! 🐾"
    else:
        return "I'm trained mostly on dog breeds! Try asking about Golden Retrievers, Pugs, or Huskies, or upload a photo in the Lens tab!"

# ==========================================
# 5. MAIN UI LAYOUT
# ==========================================

# HEADER
st.markdown("""
    <div class="hero-container">
        <div class="lens-logo">🐶</div>
        <h1 style="margin:0; font-size: 24px;">Dog Lens & Chat</h1>
    </div>
""", unsafe_allow_html=True)

# TABS FOR NAVIGATION
tab_lens, tab_chat = st.tabs(["📸 Lens Scanner", "💬 Ask AI"])

# ------------------------------------------------------------------
# TAB 1: LENS SCANNER (The Google Lens UI)
# ------------------------------------------------------------------
with tab_lens:
    if st.session_state.page == 'HOME':
        st.markdown('<div style="background:white; padding:30px; border-radius:24px; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.05);">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a dog photo", type=['jpg','jpeg','png'])
        st.caption("Upload to identify breed instantly")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            with st.spinner("Analyzing neural patterns..."):
                import tensorflow as tf # Lazy load
                
                if not os.path.exists(MODEL_PATH):
                    st.error("Model missing! Please run training script.")
                    st.stop()
                    
                model = tf.keras.models.load_model(MODEL_PATH)
                with open(CLASSES_PATH, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]

                image = Image.open(uploaded_file).convert('RGB')
                img_resized = ImageOps.fit(image, (224, 224), Image.LANCZOS)
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                
                preds = model.predict(img_array)
                score = preds[0]
                top_idx = np.argmax(score)
                conf = 100 * np.max(score)
                raw_name = classes[top_idx]
                breed_name = raw_name.split('-', 1)[1].replace('_', ' ').title() if '-' in raw_name else raw_name.replace('_', ' ').title()

                alts = [{"name": classes[i].split('-', 1)[1].replace('_', ' ').title() if '-' in classes[i] else classes[i], "conf": 100*score[i]} for i in np.argsort(score)[-3:][::-1]]

                st.session_state.result = {"image": image, "breed": breed_name, "conf": conf, "alts": alts}
                st.session_state.page = 'RESULT'
                st.rerun()

    elif st.session_state.page == 'RESULT':
        data = st.session_state.result
        st.image(data['image'], use_container_width=True)
        
        st.markdown('<div class="result-sheet">', unsafe_allow_html=True)
        st.markdown('<div style="width:40px; height:4px; background:#e0e0e0; border-radius:10px; margin:0 auto 20px;"></div>', unsafe_allow_html=True)
        
        if data['conf'] < CONFIDENCE_THRESHOLD:
            st.warning(f"Low Confidence ({data['conf']:.1f}%). Possibly not a dog.")
            with st.expander("See guesses"):
                for alt in data['alts']: st.write(f"{alt['name']}: {alt['conf']:.1f}%")
        else:
            info = get_details(data['breed'])
            st.markdown(f"<h2 class='breed-title'>{data['breed']}</h2>", unsafe_allow_html=True)
            st.markdown(f"<span style='background:#e8f0fe; color:#1a73e8; padding:5px 12px; border-radius:15px; font-weight:bold; font-size:14px;'>Match: {data['conf']:.1f}%</span>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:15px; margin-top:20px;">
                <div style="background:#f8f9fa; padding:15px; border-radius:15px;">
                    <div style="font-size:11px; color:#5f6368; font-weight:bold;">ORIGIN</div>
                    <div style="font-weight:500;">{info['Origin']}</div>
                </div>
                <div style="background:#f8f9fa; padding:15px; border-radius:15px;">
                    <div style="font-size:11px; color:#5f6368; font-weight:bold;">LIFESPAN</div>
                    <div style="font-weight:500;">{info['Life Span']}</div>
                </div>
                <div style="background:#f8f9fa; padding:15px; border-radius:15px; grid-column:span 2;">
                    <div style="font-size:11px; color:#5f6368; font-weight:bold;">TEMPERAMENT</div>
                    <div style="font-weight:500;">{info['Temperament']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💬 Ask Chatbot about this breed"):
                # Switch to chat and prompt about this breed
                st.session_state.messages.append({"role": "user", "content": f"Tell me about the {data['breed']}"})
                st.session_state.messages.append({"role": "assistant", "content": chat_response(f"Tell me about the {data['breed']}")})
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("⬅ Scan New"): reset()

# ------------------------------------------------------------------
# TAB 2: AI CHATBOT (The Chat UI)
# ------------------------------------------------------------------
with tab_chat:
    st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)
    
    # Display Chat History
    for message in st.session_state.messages:
        role_class = "user-msg" if message["role"] == "user" else "bot-msg"
        align = "right" if message["role"] == "user" else "left"
        
        st.markdown(f"""
        <div style="display: flex; justify-content: {align};">
            <div class="chat-bubble {role_class}">
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Chat Input
    if prompt := st.chat_input("Ask about diets, breeds, or training..."):
        # 1. Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun() # Rerun to show user message immediately

# Handle Chat Response generation (After Rerun)
if st.session_state.messages[-1]["role"] == "user":
    user_text = st.session_state.messages[-1]["content"]
    
    with st.spinner("Thinking..."):
        time.sleep(0.6) # Natural delay
        response_text = chat_response(user_text)
        
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()
