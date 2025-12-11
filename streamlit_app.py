import streamlit as st
import numpy as np
import os
from PIL import Image, ImageOps

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PawIdentify",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CSS STYLING (For Colors & Cards only)
# ==========================================
st.markdown("""
<style>
    /* GLOBAL SETTINGS */
    .stApp {
        background-color: #F4F6F9;
        font-family: 'Helvetica Neue', sans-serif;
    }
    header, footer, #MainMenu {visibility: hidden;}

    /* CUSTOM BORDERS & CARDS */
    div[data-testid="stVerticalBlock"] > div {
        border-radius: 15px;
    }
    
    /* CHAT BUBBLES */
    .user-chat {
        background-color: #007AFF;
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        text-align: right;
        display: inline-block;
        float: right;
    }
    .bot-chat {
        background-color: #E9E9EB;
        color: black;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        display: inline-block;
    }
    
    /* UPLOAD BOX STYLING */
    [data-testid="stFileUploader"] {
        padding: 30px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. ROBUST DATABASE
# ==========================================
BREED_DB = {
    "German Shepherd": {
        "Origin": "Germany", "Life": "7-10 yrs", "Group": "Herding",
        "Diet": "High protein (22%+) for muscle maintenance.",
        "Bio": "Confident, courageous, and smart. A loyal guardian."
    },
    "Golden Retriever": {
        "Origin": "Scotland", "Life": "10-12 yrs", "Group": "Sporting",
        "Diet": "Balanced diet. Watch calories to prevent obesity.",
        "Bio": "Friendly, intelligent, and devoted. Loves water."
    },
    "Labrador Retriever": {
        "Origin": "Canada", "Life": "10-12 yrs", "Group": "Sporting",
        "Diet": "Strict portion control. They are prone to overeating.",
        "Bio": "Outgoing, even-tempered, and gentle. Great family dog."
    },
    "Siberian Husky": {
        "Origin": "Siberia", "Life": "12-14 yrs", "Group": "Working",
        "Diet": "Fish-based diet rich in Omega-3/6 fatty acids.",
        "Bio": "Loyal, outgoing, and mischievous. Born to run."
    },
    "Pug": {
        "Origin": "China", "Life": "13-15 yrs", "Group": "Toy",
        "Diet": "Low-calorie food. Prone to rapid weight gain.",
        "Bio": "Charming, mischievous, and loving."
    },
    "Chihuahua": {
        "Origin": "Mexico", "Life": "14-16 yrs", "Group": "Toy",
        "Diet": "Nutrient-dense small kibble.",
        "Bio": "Graceful, sassy, and devoted to one person."
    },
     "Rottweiler": {
        "Origin": "Germany", "Life": "9-10 yrs", "Group": "Working",
        "Diet": "High protein. Needs joint supplements.",
        "Bio": "A robust working breed. Loyal and confident."
    },
     "Beagle": {
        "Origin": "UK", "Life": "10-15 yrs", "Group": "Hound",
        "Diet": "Measured meals. They will eat anything.",
        "Bio": "Merry, curious, and friendly. Driven by scent."
    }
}

FALLBACK = {
    "Origin": "Unknown", "Life": "10-13 yrs", "Group": "Mixed/Unknown",
    "Diet": "Standard balanced dog food.",
    "Bio": "A loyal canine companion."
}

# ==========================================
# 4. LOGIC & FUNCTIONS
# ==========================================
MODEL_PATH = 'final_model.keras'
CLASSES_PATH = 'classes.txt'
CONFIDENCE_THRESHOLD = 50.0

if 'page' not in st.session_state: st.session_state.page = 'HOME'
if 'data' not in st.session_state: st.session_state.data = None
if 'chat' not in st.session_state: st.session_state.chat = []

def get_info(name):
    for key in BREED_DB:
        if key in name: return BREED_DB[key]
    return FALLBACK

def smart_chat(breed, text):
    """
    Fixed logic to distinguish between 'Live in' (Origin) and 'Live long' (Lifespan)
    """
    text = text.lower()
    info = get_info(breed)
    
    # 1. DIET
    if any(x in text for x in ["food", "diet", "eat", "feed"]):
        return f"🍖 **Diet:** {info['Diet']}"
    
    # 2. ORIGIN (Fixing the overlap)
    # Checks for 'where', 'from', 'country', 'origin' OR 'live' without 'long'
    elif any(x in text for x in ["origin", "from", "country", "where"]):
        return f"🌍 **Origin:** The {breed} originates from {info['Origin']}."
    elif "live" in text and "long" not in text and "years" not in text:
         return f"🌍 **Origin:** They are originally from {info['Origin']}."

    # 3. LIFESPAN
    elif any(x in text for x in ["life", "age", "old", "die", "long", "years"]):
        return f"⏳ **Lifespan:** The {breed} typically lives for {info['Life']}."
    
    # 4. DEFAULT
    else:
        return f"The **{breed}** is a {info['Group']} dog. {info['Bio']} Ask me about their diet!"

def reset():
    st.session_state.page = 'HOME'
    st.session_state.data = None
    st.session_state.chat = []
    st.rerun()

# ==========================================
# 5. UI: HOME SCREEN
# ==========================================
if st.session_state.page == 'HOME':
    
    # Hero Header
    st.markdown("<h1 style='text-align: center; color: #1F2937;'>🐾 PawIdentify</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280; font-size: 18px;'>AI Dog Breed Detector & Expert Chatbot</p>", unsafe_allow_html=True)
    st.write("---")

    # Center the upload button using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("Upload a dog photo...", type=['jpg','png','jpeg'])

    if uploaded_file:
        with st.spinner("Analyzing neural features..."):
            import tensorflow as tf
            
            if not os.path.exists(MODEL_PATH):
                st.error("Model missing. Please train first.")
                st.stop()
            
            # Load Model
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASSES_PATH, 'r') as f: classes = [x.strip() for x in f.readlines()]
            
            # Process Image
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
            
            # Save State
            st.session_state.data = {"image": image, "breed": breed_name, "conf": conf}
            st.session_state.chat = [{"role": "bot", "content": f"Woof! I see a **{breed_name}**. Ask me about their diet, origin, or lifespan!"}]
            st.session_state.page = 'RESULT'
            st.rerun()

# ==========================================
# 6. UI: RESULT DASHBOARD (Split View)
# ==========================================
elif st.session_state.page == 'RESULT':
    data = st.session_state.data
    breed = data['breed']
    conf = data['conf']
    
    # --- FAILURE CHECK ---
    if conf < CONFIDENCE_THRESHOLD:
        st.markdown(f"""
        <div style="background: #FEF2F2; padding: 30px; border-radius: 20px; text-align: center; border: 1px solid #FCA5A5;">
            <h1 style="color: #991B1B; margin:0;">⚠️ No Dog Detected</h1>
            <p style="font-size: 18px; margin-top: 10px;">Confidence: {conf:.1f}%</p>
            <p>Our AI isn't confident this is a dog. Please try a clearer photo.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        if st.button("⬅ Try Again", type="primary"): reset()
        
    # --- SUCCESS DASHBOARD ---
    else:
        info = get_info(breed)
        
        # SPLIT SCREEN: LEFT (Profile) | RIGHT (Chat)
        left, right = st.columns([1, 1.3], gap="large")
        
        # --- LEFT SIDE: THE ID CARD ---
        with left:
            st.image(data['image'], use_container_width=True)
            
            # Use Native Streamlit Elements (Safe, no raw HTML bugs)
            st.header(breed)
            st.caption(f"Match Confidence: {conf:.1f}%")
            
            st.info(f"**Bio:** {info['Bio']}")
            
            # Metric Grid
            m1, m2 = st.columns(2)
            m1.metric("🌍 Origin", info['Origin'])
            m2.metric("⏳ Lifespan", info['Life'])
            
            m3, m4 = st.columns(2)
            m3.metric("🐕 Group", info['Group'])
            m4.metric("🥩 Diet Type", "See Chat ➜")
            
            st.write("---")
            if st.button("📸 Scan New Dog", use_container_width=True): reset()

        # --- RIGHT SIDE: THE CHATBOT ---
        with right:
            st.subheader(f"💬 Chat about {breed}s")
            
            # Chat Container (Scrollable)
            chat_container = st.container(height=500)
            
            with chat_container:
                for msg in st.session_state.chat:
                    if msg['role'] == 'user':
                        # Render user bubble (Right aligned)
                        st.markdown(f"<div style='text-align: right;'><span class='user-chat'>{msg['content']}</span></div><div style='clear: both;'></div>", unsafe_allow_html=True)
                    else:
                        # Render bot bubble (Left aligned)
                        st.markdown(f"<div style='text-align: left;'><span class='bot-chat'>{msg['content']}</span></div><div style='clear: both;'></div>", unsafe_allow_html=True)

            # Chat Input
            if prompt := st.chat_input(f"Ask about {breed}..."):
                # 1. User Message
                st.session_state.chat.append({"role": "user", "content": prompt})
                
                # 2. Generate Answer
                reply = smart_chat(breed, prompt)
                
                # 3. Bot Message
                st.session_state.chat.append({"role": "bot", "content": reply})
                st.rerun()
