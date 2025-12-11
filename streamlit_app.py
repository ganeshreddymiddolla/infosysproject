import streamlit as st
import numpy as np
import time
import os
import random
from PIL import Image, ImageOps

# ==========================================
# 1. PAGE CONFIG (Wide Mode for Split View)
# ==========================================
st.set_page_config(
    page_title="PawPedia AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. MODERN UI DESIGN (CSS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    /* GLOBAL RESET */
    .stApp {
        background-color: #F3F4F6;
        font-family: 'DM Sans', sans-serif;
        color: #1F2937;
    }
    header, footer, #MainMenu {visibility: hidden;}

    /* --- LANDING PAGE --- */
    .upload-container {
        background: white;
        padding: 40px;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        text-align: center;
        max-width: 600px;
        margin: 50px auto;
        border: 1px solid #E5E7EB;
    }
    .brand-title {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    /* --- RESULT CARD (Left Side) --- */
    .profile-card {
        background: white;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
        height: 100%;
    }
    .breed-header {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        margin-top: 15px;
        margin-bottom: 5px;
    }
    .stat-badge {
        background: #EEF2FF;
        color: #4F46E5;
        padding: 6px 12px;
        border-radius: 100px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 20px;
    }
    .detail-row {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #F3F4F6;
    }
    .detail-label { color: #6B7280; font-size: 14px; }
    .detail-val { font-weight: 500; color: #111827; font-size: 14px; text-align: right; max-width: 60%; }

    /* --- CHAT INTERFACE (Right Side) --- */
    .chat-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 20px;
        border: 1px solid #E5E7EB;
        height: 600px;
        display: flex;
        flex-direction: column;
    }
    .chat-header {
        font-weight: 700;
        color: #374151;
        margin-bottom: 15px;
        padding-bottom: 15px;
        border-bottom: 1px solid #F3F4F6;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 16px;
        margin-bottom: 10px;
        font-size: 14px;
        line-height: 1.5;
        max-width: 85%;
        animation: fadeIn 0.3s ease;
    }
    .bot-msg {
        background: #F3F4F6;
        color: #1F2937;
        border-bottom-left-radius: 4px;
    }
    .user-msg {
        background: #4F46E5;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    /* ANIMATIONS */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

    /* BUTTONS */
    .stButton > button {
        background-color: #4F46E5;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover { background-color: #4338CA; }
    
    /* ERROR STATE */
    .error-card {
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. ADVANCED BREED DATABASE (The "Brain")
# ==========================================
BREED_DATA = {
    "Golden Retriever": {
        "Origin": "Scotland", "Life": "10-12 yrs", "Group": "Sporting",
        "Bio": "Intelligent, friendly, and devoted. They love water and playing fetch.",
        "Diet": "2-3 cups of high-quality dry food/day. Avoid fatty table scraps.",
        "Training": "Eager to please. Use positive reinforcement and treats.",
        "Health": "Prone to hip dysplasia. Regular vet checks are essential."
    },
    "German Shepherd": {
        "Origin": "Germany", "Life": "7-10 yrs", "Group": "Herding",
        "Bio": "Confident, courageous, and smart. The ultimate working dog.",
        "Diet": "High-protein diet (22%+) to support muscle maintenance.",
        "Training": "Requires firm, consistent leadership and mental stimulation.",
        "Health": "Watch for joint issues. Keep them active but don't over-exercise puppies."
    },
    "Labrador Retriever": {
        "Origin": "Canada", "Life": "10-12 yrs", "Group": "Sporting",
        "Bio": "Friendly and active. America's most popular dog breed.",
        "Diet": "They love to eat! Measure portions carefully to prevent obesity.",
        "Training": "Very trainable. Good for agility and obedience competitions.",
        "Health": "Prone to obesity and ear infections. Clean ears regularly."
    },
    "Siberian Husky": {
        "Origin": "Siberia", "Life": "12-14 yrs", "Group": "Working",
        "Bio": "Loyal, mischievous, and outgoing. Known for endurance.",
        "Diet": "High-protein, high-fat diet similar to their ancestral intake.",
        "Training": "Independent thinkers. Keep training sessions short and fun.",
        "Health": "Generally healthy, but watch for eye conditions."
    },
    "Pug": {
        "Origin": "China", "Life": "13-15 yrs", "Group": "Toy",
        "Bio": "Charming, mischievous, and loving. A lot of dog in a small space.",
        "Diet": "Calorie-controlled diet. They gain weight very easily.",
        "Training": "Can be stubborn. Food is a great motivator.",
        "Health": "Sensitive to heat. Clean facial wrinkles daily to prevent infection."
    },
    # Fallback for other breeds
    "default": {
        "Origin": "Unknown", "Life": "10-13 yrs", "Group": "Companion",
        "Bio": "A loyal companion dog.",
        "Diet": "Balanced high-quality dog food appropriate for their size.",
        "Training": "Consistent positive reinforcement works best.",
        "Health": "Regular checkups and vaccinations are key."
    }
}

# ==========================================
# 4. LOGIC ENGINE
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
    # Fuzzy match breed name to database
    for key in BREED_DATA:
        if key in name: return BREED_DATA[key]
    return BREED_DATA["default"]

def smart_chat_response(breed, question):
    """
    Generates a context-aware answer based on the identified breed.
    """
    question = question.lower()
    info = get_breed_info(breed)
    
    # 1. DIET QUESTIONS
    if any(x in question for x in ["eat", "food", "diet", "feed", "hungry"]):
        return f"🍖 **Diet Advice for {breed}s:** {info['Diet']}"
    
    # 2. TRAINING QUESTIONS
    elif any(x in question for x in ["train", "teach", "sit", "stay", "behave"]):
        return f"🎓 **Training Tip:** {info['Training']}"
    
    # 3. HEALTH QUESTIONS
    elif any(x in question for x in ["health", "sick", "doctor", "vet", "care"]):
        return f"❤️ **Health Note:** {info['Health']}"
    
    # 4. ORIGIN/BIO
    elif any(x in question for x in ["origin", "from", "history", "who"]):
        return f"🌍 The {breed} originates from {info['Origin']}. {info['Bio']}"
    
    # 5. GENERAL / GREETING
    elif any(x in question for x in ["hi", "hello", "hey"]):
        return f"Woof! Ask me anything about taking care of this {breed}!"
    
    # 6. FALLBACK
    else:
        return f"That's a great question about the {breed}. While I focus on diet, training, and health, generally they are {info['Group']} dogs known for being {info['Bio'].split('.')[0].lower()}."

# ==========================================
# 5. UI: LANDING PAGE
# ==========================================
if st.session_state.page == 'HOME':
    
    st.markdown("""
        <div class='upload-container'>
            <div style='font-size: 60px;'>🐾</div>
            <h1 class='brand-title'>PawPedia AI</h1>
            <p style='color:#6B7280; margin-bottom: 30px;'>
                Upload a photo to identify the breed and chat with a specialized AI expert.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Centered Upload Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(" ", type=['jpg','png','jpeg'], label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("Scanning features..."):
            # Lazy Import
            import tensorflow as tf
            
            # Load Model
            if not os.path.exists(MODEL_PATH):
                st.error("Model file missing.")
                st.stop()
            
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASSES_PATH, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            
            # Process
            image = Image.open(uploaded_file).convert('RGB')
            img_resized = ImageOps.fit(image, (224, 224), Image.LANCZOS)
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            preds = model.predict(img_array)
            score = preds[0]
            top_idx = np.argmax(score)
            conf = 100 * np.max(score)
            
            raw_name = classes[top_idx]
            breed_name = raw_name.split('-', 1)[1].replace('_', ' ').title() if '-' in raw_name else raw_name
            
            # Store State
            st.session_state.data = {
                "image": image,
                "breed": breed_name,
                "conf": conf
            }
            
            # Initialize Chat with Greeting
            st.session_state.chat_history = [{
                "role": "assistant", 
                "content": f"I've identified this as a **{breed_name}**! I'm an expert on this breed. Ask me about their diet, training, or health."
            }]
            
            st.session_state.page = 'RESULT'
            st.rerun()

# ==========================================
# 6. UI: RESULT & CHAT DASHBOARD
# ==========================================
elif st.session_state.page == 'RESULT':
    data = st.session_state.data
    breed = data['breed']
    conf = data['conf']
    
    # --- LOGIC: CHECK CONFIDENCE ---
    if conf < CONFIDENCE_THRESHOLD:
        # FAILED STATE
        st.markdown(f"""
            <div class='upload-container'>
                <div style='font-size: 50px;'>⚠️</div>
                <h2 style='color:#991B1B;'>No Dog Found</h2>
                <p>Confidence: {conf:.1f}%</p>
                <div class='error-card'>
                    Our AI isn't confident this is a dog. It might be a human, object, or unclear photo.
                </div>
            </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button("Try Another Photo"): reset_app()
            
    else:
        # SUCCESS STATE - SPLIT VIEW
        info = get_breed_info(breed)
        
        # Create Layout: Left (Profile) | Right (Chat)
        col_profile, col_chat = st.columns([1, 1.2], gap="large")
        
        # --- LEFT COLUMN: BREED PROFILE ---
        with col_profile:
            st.image(data['image'], use_container_width=True)
            
            st.markdown(f"""
            <div class='profile-card'>
                <div class='stat-badge'>Match: {conf:.1f}%</div>
                <h2 class='breed-header'>{breed}</h2>
                <p style='color:#6B7280; font-size:14px; margin-bottom:20px;'>{info['Bio']}</p>
                
                <div class='detail-row'>
                    <span class='detail-label'>Origin</span>
                    <span class='detail-val'>{info['Origin']}</span>
                </div>
                <div class='detail-row'>
                    <span class='detail-label'>Lifespan</span>
                    <span class='detail-val'>{info['Life']}</span>
                </div>
                <div class='detail-row'>
                    <span class='detail-label'>Group</span>
                    <span class='detail-val'>{info['Group']}</span>
                </div>
                <br>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("⬅ Scan New Dog"): reset_app()

        # --- RIGHT COLUMN: CONTEXTUAL CHATBOT ---
        with col_chat:
            st.markdown(f"""
            <div class='chat-header'>
                <span>💬</span> Chat with {breed} Expert
            </div>
            """, unsafe_allow_html=True)
            
            # Chat History Container
            chat_container = st.container(height=400)
            
            with chat_container:
                for msg in st.session_state.chat_history:
                    css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
                    align = "right" if msg["role"] == "user" else "left"
                    st.markdown(f"""
                        <div style='display:flex; justify-content:{align};'>
                            <div class='chat-bubble {css_class}'>{msg["content"]}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Chat Input Area
            if prompt := st.chat_input(f"Ask about {breed}s..."):
                # 1. Append User Msg
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # 2. Generate Smart Answer
                answer = smart_chat_response(breed, prompt)
                
                # 3. Append Bot Msg
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
