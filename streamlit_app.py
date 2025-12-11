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
# 2. HIGH CONTRAST CSS (Black Text Fix)
# ==========================================
st.markdown("""
<style>
    /* GLOBAL TEXT COLOR FORCE BLACK */
    .stApp, p, h1, h2, h3, div, span, label {
        color: #000000 !important;
        font-family: 'Helvetica', sans-serif;
    }
    
    header, footer, #MainMenu {visibility: hidden;}

    /* CHAT BUBBLES - HIGH CONTRAST */
    .user-chat {
        background-color: #000000; /* Black background */
        color: #ffffff !important; /* White text */
        padding: 12px 18px;
        border-radius: 18px 18px 0 18px;
        margin: 5px 0;
        text-align: right;
        display: inline-block;
        float: right;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .bot-chat {
        background-color: #F3F4F6; /* Light Gray */
        color: #000000 !important; /* Black text */
        border: 1px solid #D1D5DB;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 0;
        margin: 5px 0;
        display: inline-block;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* UPLOAD BOX */
    [data-testid="stFileUploader"] {
        padding: 30px;
        background: white;
        border: 2px dashed #000000;
        border-radius: 20px;
    }
    
    /* METRICS */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #4B5563 !important; /* Dark Gray for labels */
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATABASE (Added Weather/Climate)
# ==========================================
BREED_DB = {
    "German Shepherd": {
        "Origin": "Germany", "Life": "7-10 yrs", "Group": "Herding",
        "Diet": "High protein (22%+) for muscle.",
        "Climate": "Tolerates cold well (Double coat). Avoid extreme heat.",
        "Bio": "Confident, courageous, and smart. A loyal guardian."
    },
    "Golden Retriever": {
        "Origin": "Scotland", "Life": "10-12 yrs", "Group": "Sporting",
        "Diet": "Balanced diet. Watch calories.",
        "Climate": "Good in cold weather. Needs shade in summer.",
        "Bio": "Friendly, intelligent, and devoted. Loves water."
    },
    "Labrador Retriever": {
        "Origin": "Canada", "Life": "10-12 yrs", "Group": "Sporting",
        "Diet": "Strict portion control (prone to obesity).",
        "Climate": "Hardy in most climates. Loves swimming.",
        "Bio": "Outgoing, even-tempered, and gentle."
    },
    "Siberian Husky": {
        "Origin": "Siberia", "Life": "12-14 yrs", "Group": "Working",
        "Diet": "High protein/fat diet.",
        "Climate": "Thrives in freezing cold. DO NOT SHAVE in summer.",
        "Bio": "Loyal, outgoing, and mischievous. Born to run."
    },
    "Pug": {
        "Origin": "China", "Life": "13-15 yrs", "Group": "Toy",
        "Diet": "Low-calorie food.",
        "Climate": "Sensitive to heat & humidity (Brachycephalic). Keep cool.",
        "Bio": "Charming, mischievous, and loving."
    },
    "Chihuahua": {
        "Origin": "Mexico", "Life": "14-16 yrs", "Group": "Toy",
        "Diet": "Nutrient-dense small kibble.",
        "Climate": "Loves heat. Needs sweaters in the cold.",
        "Bio": "Graceful, sassy, and devoted."
    },
     "Rottweiler": {
        "Origin": "Germany", "Life": "9-10 yrs", "Group": "Working",
        "Diet": "High protein. Joint supplements recommended.",
        "Climate": "Tolerates cool weather. Avoid overheating.",
        "Bio": "A robust working breed. Loyal and confident."
    },
     "Beagle": {
        "Origin": "UK", "Life": "10-15 yrs", "Group": "Hound",
        "Diet": "Measured meals. Food obsessed.",
        "Climate": "Adaptable to most climates.",
        "Bio": "Merry, curious, and friendly."
    }
}

FALLBACK = {
    "Origin": "Unknown", "Life": "10-13 yrs", "Group": "Mixed",
    "Diet": "Balanced dog food.", "Climate": "Moderate temperatures.",
    "Bio": "A loyal canine companion."
}

# ==========================================
# 4. LOGIC
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
    text = text.lower()
    words = text.split() # Split into words to avoid partial matching (e.g. wheat vs eat)
    info = get_info(breed)
    
    # 1. WEATHER / CLIMATE
    if any(w in text for w in ["weather", "climate", "hot", "cold", "temp", "winter", "summer"]):
        return f"☀️ **Climate:** {info['Climate']}"

    # 2. DIET (Fixed 'Wheat' bug)
    elif any(w in words for w in ["food", "diet", "eat", "feed"]):
        return f"🍖 **Diet:** {info['Diet']}"
    
    # 3. ORIGIN
    elif any(w in text for w in ["origin", "from", "country", "where"]):
        return f"🌍 **Origin:** The {breed} originates from {info['Origin']}."

    # 4. LIFESPAN
    elif any(w in text for w in ["life", "age", "old", "die", "long", "years"]):
        return f"⏳ **Lifespan:** The {breed} typically lives for {info['Life']}."
    
    # 5. DEFAULT
    else:
        return f"The **{breed}** is a {info['Group']} dog. {info['Bio']} Ask about their diet or weather preference!"

def reset():
    st.session_state.page = 'HOME'
    st.session_state.data = None
    st.session_state.chat = []
    st.rerun()

# ==========================================
# 5. HOME SCREEN
# ==========================================
if st.session_state.page == 'HOME':
    
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🐾 PawIdentify</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>AI Dog Breed Detector</p>", unsafe_allow_html=True)
    st.write("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("Upload a dog photo...", type=['jpg','png','jpeg'])

    if uploaded_file:
        with st.spinner("Analyzing neural features..."):
            import tensorflow as tf
            
            if not os.path.exists(MODEL_PATH):
                st.error("Model missing. Please train first.")
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
            breed_name = raw_name.split('-', 1)[1].replace('_', ' ').title() if '-' in raw_name else raw_name
            
            st.session_state.data = {"image": image, "breed": breed_name, "conf": conf}
            st.session_state.chat = [{"role": "bot", "content": f"Woof! I see a **{breed_name}**. Ask me about their diet, origin, or climate!"}]
            st.session_state.page = 'RESULT'
            st.rerun()

# ==========================================
# 6. RESULT DASHBOARD
# ==========================================
elif st.session_state.page == 'RESULT':
    data = st.session_state.data
    breed = data['breed']
    conf = data['conf']
    
    if conf < CONFIDENCE_THRESHOLD:
        st.markdown(f"""
        <div style="background: #FFE4E6; padding: 30px; border-radius: 20px; text-align: center; border: 2px solid #E11D48;">
            <h1 style="color: #BE123C; margin:0;">⚠️ No Dog Detected</h1>
            <p style="font-size: 18px; margin-top: 10px; color: black;">Confidence: {conf:.1f}%</p>
            <p style="color: black;">Our AI isn't confident this is a dog. Please try a clearer photo.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Try Again"): reset()
        
    else:
        info = get_info(breed)
        
        # SPLIT SCREEN
        left, right = st.columns([1, 1.3], gap="large")
        
        # --- LEFT: ID CARD ---
        with left:
            st.image(data['image'], use_container_width=True)
            
            st.header(breed)
            st.markdown(f"**Match Confidence:** {conf:.1f}%")
            
            st.markdown(f"**Bio:** {info['Bio']}")
            st.divider()
            
            c1, c2 = st.columns(2)
            c1.metric("🌍 Origin", info['Origin'])
            c2.metric("⏳ Lifespan", info['Life'])
            
            c3, c4 = st.columns(2)
            c3.metric("🐕 Group", info['Group'])
            c4.metric("☀️ Climate", "Ask Chat")
            
            st.write("")
            if st.button("📸 Scan New Dog", use_container_width=True): reset()

        # --- RIGHT: CHATBOT ---
        with right:
            st.subheader(f"💬 Chat about {breed}s")
            
            chat_container = st.container(height=500)
            
            with chat_container:
                for msg in st.session_state.chat:
                    if msg['role'] == 'user':
                        st.markdown(f"<div style='text-align: right;'><span class='user-chat'>{msg['content']}</span></div><div style='clear: both;'></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align: left;'><span class='bot-chat'>{msg['content']}</span></div><div style='clear: both;'></div>", unsafe_allow_html=True)

            if prompt := st.chat_input(f"Ask about {breed}..."):
                st.session_state.chat.append({"role": "user", "content": prompt})
                reply = smart_chat(breed, prompt)
                st.session_state.chat.append({"role": "bot", "content": reply})
                st.rerun()
