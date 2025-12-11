# PawIdentify Pro — Redesigned UI (Light Theme)
# Single-file Streamlit app — professional, attractive, easy to use
# Author: AI Assistant (redesign)
# Version: 3.1.0 (UI Redesign)

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
import tensorflow as tf
import time

# -----------------------------
# App config
# -----------------------------
APP_TITLE = "PawIdentify Pro"
APP_ICON = "🐕"
LAYOUT_MODE = "wide"
MODEL_FILE_PATH = "final_model.keras"
CLASSES_FILE_PATH = "classes.txt"
CONFIDENCE_THRESHOLD = 50.0  # percent
IMG_WIDTH = 224
IMG_HEIGHT = 224

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=LAYOUT_MODE, initial_sidebar_state="auto")

# -----------------------------
# Light theme CSS — clean, modern, accessible
# -----------------------------
def inject_light_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
            background: #ffffff;
            color: #0f172a;
        }
        /* remove default header/footer */
        #MainMenu {visibility: hidden}
        footer {visibility: hidden}
        header {visibility: hidden}

        /* block container padding */
        .block-container{ padding: 1.5rem 2rem 3rem 2rem; }

        /* Brand */
        .brand { text-align:center; padding-top:8px; margin-bottom:10px; }
        .brand h1{ font-size:44px; margin: 6px 0 0 0; font-weight:800; background: linear-gradient(90deg,#2563eb,#7c3aed); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .brand p{ margin:0; color:#475569; }

        /* Upload card */
        .upload-card{ background:#FAFBFF; border:1px solid #E6EEF8; border-radius:16px; padding:20px; }

        /* Left profile card */
        .profile-card{ background:#FFFFFF; border:1px solid #EEF2FF; border-radius:16px; padding:18px; box-shadow:0 8px 26px rgba(15,23,42,0.04); }
        .confidence-pill{ display:inline-block; padding:6px 12px; border-radius:999px; background:linear-gradient(90deg,#dcfce7,#bbf7d0); color:#065f46; font-weight:700; }

        /* Chat container */
        .chat-container{ background:#FFFFFF; border:1px solid #EEF2FF; border-radius:16px; padding:14px; height:560px; display:flex; flex-direction:column; }
        .chat-history{ overflow:auto; padding-bottom:10px; flex:1; }
        .chat-user{ align-self:flex-end; background:#2563EB; color:white; padding:10px 14px; border-radius:14px; margin:8px 0; max-width:80%; }
        .chat-bot{ align-self:flex-start; background:#F1F5F9; color:#0f172a; padding:10px 14px; border-radius:14px; margin:8px 0; max-width:80%; border:1px solid #E6EEF8; }

        /* Buttons */
        .stButton>button { background:#2563EB; color:white; border-radius:12px; padding:10px 16px; font-weight:600; }
        .stButton>button:hover{ background:#1e40af }

        /* small screens adjustments */
        @media (max-width: 700px){
            .brand h1{ font-size:32px }
            .chat-container{ height:420px }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_light_css()

# -----------------------------
# Data: simplified knowledge base (kept from original but compact)
# -----------------------------
BREED_KB = {
    "Siberian Husky": {"Origin":"Siberia","Life Span":"12-14 years","Group":"Working","Height":"20-23 inches","Diet":"High-protein, fish is great.","Climate":"Cold-loving; shade/A/C in summer.","Grooming":"Heavy shedder; daily brushing.","Training":"Intelligent but stubborn."},
    "Golden Retriever": {"Origin":"Scotland","Life Span":"10-12 years","Group":"Sporting","Height":"21-24 inches","Diet":"Balanced diet; watch weight.","Climate":"Adaptable; loves water.","Grooming":"Regular brushing.","Training":"Eager to please."},
    "Labrador Retriever": {"Origin":"Canada","Life Span":"10-12 years","Group":"Sporting","Height":"21-24 inches","Diet":"Portion control.","Climate":"Water-resistant coat.","Grooming":"Weekly brush.","Training":"Friendly and trainable."},
    "Poodle": {"Origin":"France/Germany","Life Span":"10-18 years","Group":"Non-Sporting","Height":"Varies","Diet":"Omega rich food.","Climate":"Adaptable.","Grooming":"Professional grooming every 4-6 weeks.","Training":"Very intelligent."},
    "Fallback": {"Origin":"International","Life Span":"10-13 years","Group":"Mixed","Height":"Varies","Diet":"Balanced diet appropriate for size","Climate":"Moderate","Grooming":"Regular brushing","Training":"Positive reinforcement"}
}

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource
def load_model():
    """Load keras model and classes file if available. Returns (model, classes) or (None, None)."""
    if not os.path.exists(MODEL_FILE_PATH):
        return None, None
    try:
        model = tf.keras.models.load_model(MODEL_FILE_PATH)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None
    classes = []
    if os.path.exists(CLASSES_FILE_PATH):
        try:
            with open(CLASSES_FILE_PATH, 'r') as f:
                classes = [l.strip() for l in f.readlines() if l.strip()]
        except Exception:
            classes = []
    return model, classes


def preprocess(img: Image.Image):
    img = img.convert('RGB')
    img = ImageOps.fit(img, (IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, 0)
    return arr


def lookup_breed(breed_name: str):
    if not breed_name: return BREED_KB['Fallback']
    if breed_name in BREED_KB: return BREED_KB[breed_name]
    # try partial
    for k in BREED_KB:
        if k.lower() in breed_name.lower() or breed_name.lower() in k.lower():
            return BREED_KB[k]
    return BREED_KB['Fallback']


def simple_chat(breed: str, message: str):
    msg = message.lower()
    info = lookup_breed(breed)
    if any(w in msg for w in ["food","eat","diet"]):
        return f"🍖 Diet: {info.get('Diet') }"
    if any(w in msg for w in ["origin","from","where","history"]):
        return f"🌍 Origin: {info.get('Origin')}"
    if any(w in msg for w in ["life","lifespan","age"]):
        return f"⏳ Lifespan: {info.get('Life Span')}"
    if any(w in msg for w in ["groom","brush","shed","coat"]):
        return f"🛁 Grooming: {info.get('Grooming', 'Regular brushing recommended.') }"
    if any(w in msg for w in ["train","behavior","smart"]):
        return f"🎓 Training: {info.get('Training', 'Positive reinforcement works best.') }"
    return f"📘 Quick fact: {info.get('Origin')} — {info.get('Life Span')} — {info.get('Group') }"

# -----------------------------
# Session state
# -----------------------------
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'analysis' not in st.session_state: st.session_state.analysis = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

# -----------------------------
# Layout: Header
# -----------------------------
with st.container():
    st.markdown("<div class='brand'>", unsafe_allow_html=True)
    st.markdown(f"<img src='https://cdn-icons-png.flaticon.com/512/194/194279.png' width='64' style='opacity:0.95'>", unsafe_allow_html=True)
    st.markdown(f"<h1>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown("<p>Professional, fast, and friendly — breed identification + context-aware care assistant.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")

# -----------------------------
# Home: Upload card
# -----------------------------
def home_view():
    left, right = st.columns([1, 1])
    with left:
        st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
        st.subheader("Upload a clear photo of the dog")
        st.caption("Ideal: face + chest visible, good lighting, minimal blur")
        uploaded = st.file_uploader("Drop an image or click to browse", type=['jpg','jpeg','png'])
        st.write("")
        if uploaded:
            img = Image.open(uploaded)
            st.session_state.upload_preview = img
            # quick preview
            st.image(img, use_column_width=True, caption='Uploaded image — preview')
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔎 Analyze Now"):
                # analyze
                with st.spinner("Analyzing — please wait..."):
                    model, classes = load_model()
                    if model is None or not classes:
                        st.error("Model or classes not found. Place 'final_model.keras' and 'classes.txt' next to this app.")
                        return
                    arr = preprocess(img)
                    preds = model.predict(arr)
                    top = np.argmax(preds[0])
                    conf = float(np.max(preds[0])) * 100.0
                    raw_name = classes[top] if top < len(classes) else f"Unknown-{top}"
                    if '-' in raw_name:
                        breed = raw_name.split('-',1)[1].replace('_',' ').title()
                    else:
                        breed = raw_name.title()
                    st.session_state.analysis = { 'image': img, 'breed': breed, 'confidence': conf }
                    st.session_state.chat_history = [ { 'role':'bot', 'content': f"Hi! I detected: **{breed}** ({conf:.1f}% confidence). Ask me about food, care, training." } ]
                    st.session_state.page = 'result'
                    st.experimental_rerun()
        else:
            st.info("No image uploaded yet — try a close, well-lit photo for best results.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    with right:
        st.subheader("Model & Tips")
        st.write("**Model:**** TensorFlow Keras — lightweight inference optimized for speed.")
        st.metric(label="Confidence Threshold", value=f">= {CONFIDENCE_THRESHOLD:.0f}% for 'Match'")
        st.write("**Photo tips:**")
        st.write("• Use natural light; avoid heavy shadows.")
        st.write("• Show the dog's face and chest.")
        st.write("• Avoid cluttered backgrounds.")

home_view()

# -----------------------------
# Result view: profile + chat
# -----------------------------
def result_view():
    data = st.session_state.analysis
    if data is None:
        st.error("No analysis available — return home and upload an image.")
        if st.button("Go Home"):
            st.session_state.page = 'home'
            st.experimental_rerun()
        return

    breed = data['breed']
    conf = data['confidence']
    img = data['image']

    # Two-column layout
    col_left, col_right = st.columns([1.0, 1.4], gap='large')

    with col_left:
        st.markdown("<div class='profile-card'>", unsafe_allow_html=True)
        st.image(img, use_column_width=True)
        st.markdown(f"<div style='margin-top:10px'><span class='confidence-pill'>Match: {conf:.1f}%</span></div>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='margin:8px 0 0 0'>{breed}</h2>", unsafe_allow_html=True)
        info = lookup_breed(breed)
        st.write(f"**Group:** {info.get('Group','Unknown')}")
        st.write(f"**Origin:** {info.get('Origin','Unknown')}")
        st.write(f"**Lifespan:** {info.get('Life Span','Unknown')}")
        st.write(f"**Grooming:** {info.get('Grooming','-')}")
        st.write(f"**Diet:** {info.get('Diet','-')}")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬅ Scan Another Dog"):
            st.session_state.page = 'home'
            st.session_state.analysis = None
            st.session_state.chat_history = []
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px'><div style='font-size:22px'>💬</div><div><strong>{breed} Specialist</strong><div style='font-size:12px;color:#10b981'>● Online</div></div></div>", unsafe_allow_html=True)

        # Chat history
        chat_box = st.container()
        with chat_box:
            st.markdown("<div class='chat-history' id='chat-history'>", unsafe_allow_html=True)
            for m in st.session_state.chat_history:
                if m['role'] == 'user':
                    st.markdown(f"<div class='chat-user'>{m['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-bot'>{m['content']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Input area
        st.markdown("<div style='margin-top:12px'>", unsafe_allow_html=True)
        col_inp, col_send = st.columns([4,1])
        with col_inp:
            user_msg = st.text_input("Ask about food, grooming, training...", key='chat_input')
        with col_send:
            if st.button("Send"):
                if user_msg:
                    st.session_state.chat_history.append({'role':'user','content':user_msg})
                    reply = simple_chat(breed, user_msg)
                    st.session_state.chat_history.append({'role':'bot','content':reply})
                    # clear input
                    st.session_state.chat_input = ''
                    st.experimental_rerun()
                else:
                    st.warning("Type a message before sending.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Routing
# -----------------------------
if st.session_state.page == 'home':
    # home view already rendered above, nothing else
    pass
elif st.session_state.page == 'result':
    # show result
    # If confidence below threshold, show a friendly error with retry actions
    if st.session_state.analysis and st.session_state.analysis.get('confidence', 0.0) < CONFIDENCE_THRESHOLD:
        st.warning(f"Analysis confidence is low ({st.session_state.analysis['confidence']:.1f}%). Try a clearer photo.")
        if st.button("Try Again"):
            st.session_state.page = 'home'
            st.session_state.analysis = None
            st.experimental_rerun()
    result_view()

# -----------------------------
# Footer small print
# -----------------------------
st.write("---")
colf1, colf2 = st.columns([1,3])
with colf1:
    st.write("© 2025 PawIdentify")
with colf2:
    st.caption("Model performance varies by breed and image quality. This tool is for informational purposes only and not a substitute for professional veterinary advice.")

# End of file
