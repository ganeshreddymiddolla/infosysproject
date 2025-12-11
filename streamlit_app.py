import streamlit as st
import numpy as np
import time
import os
from PIL import Image, ImageOps

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Pawdentify",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. EXPANDED KNOWLEDGE BASE (The "Brain")
# ==========================================
# This data powers both the details card and the chatbot.
BREED_KNOWLEDGE_BASE = {
    "Golden Retriever": {
        "summary": "Friendly, intelligent, and devoted. They are eager to please and make excellent family pets.",
        "diet": "High-quality kibble formulated for large, active breeds. Watch calorie intake as they love to eat.",
        "health": "Prone to hip dysplasia and certain heart issues. Regular vet checkups are essential.",
        "training": "Highly trainable and eager to learn. Positive reinforcement works best.",
        "origin": "Scotland, bred for retrieving game."
    },
    "German Shepherd": {
        "summary": "Confident, courageous, and smart. They are loyal guardians and versatile working dogs.",
        "diet": "Nutrient-dense food for high energy levels. Joint supplements are often recommended.",
        "health": "Watch for hip/elbow dysplasia and digestive issues (bloat).",
        "training": "Requires consistent, firm, and positive training. They need mental stimulation.",
        "origin": "Germany, originally for herding sheep."
    },
    "Labrador Retriever": {
        "summary": "Outgoing, even-tempered, and gentle. The quintessential family dog.",
        "diet": "They are prone to obesity. Measure food carefully and limit treats.",
        "health": "Generally healthy, but watch for joint issues and exercise-induced collapse.",
        "training": "Very intelligent and food-motivated. Training is usually easy and fun.",
        "origin": "Canada/UK, bred as fishing and retrieving dogs."
    },
    "Siberian Husky": {
        "summary": "Loyal, outgoing, and mischievous. Known for their stunning endurance and vocal nature.",
        "diet": "High-protein, high-fat diet suitable for working breeds. They have efficient metabolisms.",
        "health": "Watch for eye conditions like cataracts and hip issues.",
        "training": "Independent thinkers. Training requires patience and consistency. They are escape artists!",
        "origin": "Siberia, bred as sled dogs."
    },
    "Pug": {
        "summary": "Charming, mischievous, and loving. They live to love and be loved.",
        "diet": "Prone to rapid weight gain. Low-calorie diets and monitored portions are crucial.",
        "health": "As a brachycephalic (flat-faced) breed, they struggle in heat and can have breathing issues.",
        "training": "Can be stubborn but responds well to praise and treats.",
        "origin": "China, bred as companion dogs for nobility."
    },
     "Chihuahua": {
        "summary": "Graceful, charming, and sassy. A big personality in a tiny body.",
        "diet": "Small-breed specific formula. Because they are tiny, they need calorie-dense food frequently to prevent hypoglycemia.",
        "health": "Watch for dental issues, heart problems, and luxating patellas (knees).",
        "training": "Intelligent but can be willful. Early socialization is key to prevent excessive barking.",
        "origin": "Mexico."
    },
    # Fallback for breeds not fully detailed yet
    "Generic": {
        "summary": "A loyal companion dog.",
        "diet": "Balanced high-quality dog food appropriate for their size and age.",
        "health": "Routine vet checkups, vaccinations, and parasite prevention are key.",
        "training": "Positive reinforcement and consistency are universally effective.",
        "origin": "Various origins."
    }
}

# ==========================================
# 3. NEW MODERN CSS STYLING
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* GLOBAL RESET */
    .stApp {
        background-color: #F4F7FE; /* Soft blue-gray background */
        font-family: 'Poppins', sans-serif;
        color: #2B3674;
    }
    h1, h2, h3 { color: #2B3674; font-weight: 700; }
    p { color: #707EAE; line-height: 1.6; }
    header, footer, #MainMenu {visibility: hidden;}

    /* --- CONTAINERS & CARDS --- */
    .main-container {
        max-width: 800px;
        margin: auto;
    }
    .white-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }

    /* --- HOME SCREEN --- */
    .hero-section {
        text-align: center;
        padding: 50px 20px;
    }
    .hero-title { font-size: 2.5rem; margin-bottom: 10px; }
    .hero-subtitle { font-size: 1.1rem; color: #707EAE; }

    /* --- RESULTS SCREEN --- */
    .breed-header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .breed-name-large { font-size: 2rem; margin: 0; }
    .confidence-tag {
        background: #E6F7FF; color: #0095FF;
        padding: 8px 16px; border-radius: 30px; font-weight: 600;
    }
    .fail-tag { background: #FFE5E5; color: #D32F2F; }

    /* --- DETAILS GRID --- */
    .details-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 30px;
    }
    .detail-box {
        background: #F4F7FE;
        padding: 20px;
        border-radius: 15px;
        transition: transform 0.2s;
    }
    .detail-box:hover { transform: translateY(-3px); }
    .detail-icon { font-size: 24px; margin-bottom: 10px; }
    .detail-label { font-weight: 600; font-size: 0.9rem; color: #2B3674; }
    .detail-text { font-size: 0.9rem; color: #707EAE; margin-top: 5px; }

    /* --- CHAT INTERFACE --- */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        height: 500px;
        overflow-y: auto;
        border: 1px solid #E0E5F2;
    }
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 18px;
        margin-bottom: 12px;
        max-width: 80%;
        font-size: 14px;
        line-height: 1.5;
        animation: fadeIn 0.3s ease-in;
    }
    .user-bubble {
        background-color: #4318FF;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .bot-bubble {
        background-color: #F4F7FE;
        color: #2B3674;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

    /* --- BUTTONS --- */
    .stButton > button {
        border-radius: 15px;
        height: 50px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. BACKEND LOGIC & CHAT ENGINE
# ==========================================
MODEL_PATH = 'final_model.keras'
CLASSES_PATH = 'classes.txt'
CONFIDENCE_THRESHOLD = 50.0

# Initialize Session State
if 'page' not in st.session_state: st.session_state.page = 'HOME'
if 'result' not in st.session_state: st.session_state.result = None
# Chat history is tied to the current breed result
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

def reset_app():
    st.session_state.page = 'HOME'
    st.session_state.result = None
    st.session_state.chat_history = []
    st.rerun()

def get_breed_data(breed_name):
    # Fuzzy search for breed data
    for key in BREED_KNOWLEDGE_BASE:
        if key.lower() in breed_name.lower():
            return BREED_KNOWLEDGE_BASE[key]
    return BREED_KNOWLEDGE_BASE["Generic"]

def generate_chat_response(breed_name, prompt):
    """Generates a response based on the identified breed context."""
    prompt = prompt.lower()
    data = get_breed_data(breed_name)
    
    # Contextual answering logic
    if any(x in prompt for x in ["diet", "food", "eat", "feed"]):
        return f"🍖 **Diet Advice for {breed_name}s:**\n{data['diet']}"
    elif any(x in prompt for x in ["health", "sick", "disease", "problems"]):
        return f"🩺 **Health Considerations:**\n{data['health']}"
    elif any(x in prompt for x in ["train", "teach", "behavior"]):
        return f"🎾 **Training a {breed_name}:**\n{data['training']}"
    elif any(x in prompt for x in ["origin", "from", "history"]):
        return f"🌍 **Origin Story:**\nIt is believed the {breed_name} originated in {data['origin']}"
    elif any(x in prompt for x in ["summary", "about", "tell me"]):
        return f"🐶 **About the {breed_name}:**\n{data['summary']}"
    else:
        # General fallback if specific topic isn't found
        return f"That's an interesting question about the {breed_name}! Based on what I know, they are generally {data['summary'].lower()} Feel free to ask specifically about their diet, health, or training!"

# ==========================================
# 5. MAIN APP UI FLOW
# ==========================================

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- VIEW: HOME SCREEN ---
if st.session_state.page == 'HOME':
    st.markdown("""
        <div class="hero-section">
            <div style="font-size: 60px;">🐾</div>
            <h1 class="hero-title">Pawdentify</h1>
            <p class="hero-subtitle">Upload a photo to instantly identify the dog breed and chat about its care.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="white-card" style="text-align:center;">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a dog image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        with st.spinner("Analyzing image features..."):
            # Lazy load heavy imports
            import tensorflow as tf
            if not os.path.exists(MODEL_PATH):
                st.error("System Error: Model file not found.")
                st.stop()

            model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASSES_PATH, 'r') as f: classes = [l.strip() for l in f.readlines()]

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
            breed_name = raw_name.split('-', 1)[1].replace('_', ' ').title() if '-' in raw_name else raw_name.replace('_', ' ').title()
            
            # Get Alternatives
            top3_idx = np.argsort(score)[-3:][::-1]
            alts = [{"name": classes[i].split('-', 1)[1].replace('_', ' ').title() if '-' in classes[i] else classes[i], "conf": 100*score[i]} for i in top3_idx]

            # Save State & Transition
            st.session_state.result = {
                "image": image, "breed": breed_name, "conf": conf, "alts": alts
            }
            # Initialize chat with helpful starting message
            st.session_state.chat_history = [{
                "role": "assistant", 
                "content": f"Hello! I've identified this as a **{breed_name}**. I'm ready to answer any questions you have about their diet, health, or training!"
            }]
            st.session_state.page = 'RESULT'
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)


# --- VIEW: RESULT SCREEN ---
elif st.session_state.page == 'RESULT':
    data = st.session_state.result
    breed = data['breed']
    conf = data['conf']
    
    # --------------------------
    # SECTION 1: IMAGE & HEADER
    # --------------------------
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.image(data['image'], use_container_width=True, style="border-radius: 15px;")
    
    # LOGIC: THRESHOLD CHECK
    if conf < CONFIDENCE_THRESHOLD:
        # FAILURE CASE
        st.markdown(f"""
            <div class="breed-header-container">
                <h1 class="breed-name-large" style="color: #D32F2F;">No Dog Detected</h1>
                <div class="confidence-tag fail-tag">Low Confidence: {conf:.1f}%</div>
            </div>
            <p>We couldn't confidently identify a dog in this image. The closest visual match was {breed}, but the certainty is too low.</p>
            <p>Please try uploading a clearer picture of a dog.</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("⬅️ Try Again"): reset_app()
        
    else:
        # SUCCESS CASE
        breed_data = get_breed_data(breed)
        
        st.markdown(f"""
            <div class="breed-header-container">
                <h1 class="breed-name-large">{breed}</h1>
                <div class="confidence-tag">Create Match: {conf:.1f}%</div>
            </div>
            <p>{breed_data['summary']}</p>
        """, unsafe_allow_html=True)

        # --------------------------
        # SECTION 2: DETAILS "PIN" (Grid)
        # --------------------------
        st.markdown("""
        <div class="details-grid">
            <div class="detail-box">
                <div class="detail-icon">🌍</div>
                <div class="detail-label">Origin</div>
                <div class="detail-text">{}</div>
            </div>
            <div class="detail-box">
                 <div class="detail-icon">🍖</div>
                <div class="detail-label">Ideal Diet</div>
                <div class="detail-text">See chat for details</div>
            </div>
             <div class="detail-box">
                 <div class="detail-icon">🎾</div>
                <div class="detail-label">Training</div>
                <div class="detail-text">Responds to positive reinforcement</div>
            </div>
        </div>
        """.format(breed_data['origin']), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True) # End white card

        # --------------------------
        # SECTION 3: BREED CHATBOT
        # --------------------------
        st.markdown(f"<h3>💬 Chat about the {breed}</h3>", unsafe_allow_html=True)
        
        # Chat History Display container
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
                st.markdown(f"""
                    <div style="display: flex;">
                        <div class="chat-bubble {bubble_class}">
                            {msg["content"]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Chat Input
        if prompt := st.chat_input(f"Ask anything about {breed}s..."):
            # Add user message immediately
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()

        # Generate Response (if last message was user)
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            with st.spinner("AI is thinking..."):
                time.sleep(0.5) # UI smoothing
                response = generate_chat_response(breed, st.session_state.chat_history[-1]["content"])
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

        st.write("")
        if st.button("⬅️ Scan New Photo"): reset_app()

st.markdown('</div>', unsafe_allow_html=True) # End main container
