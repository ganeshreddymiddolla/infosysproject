"""
PawIdentify Enterprise Edition
------------------------------
A Streamlit-based AI application for Dog Breed Identification and 
Context-Aware Chatbot assistance.

Author: AI Assistant
Version: 3.0.0 (Production)
Theme: Ultra Light / Professional
"""

import streamlit as st
import numpy as np
import os
import time
from PIL import Image, ImageOps
import tensorflow as tf

# ==============================================================================
# 1. APPLICATION CONFIGURATION & CONSTANTS
# ==============================================================================

APP_TITLE = "PawIdentify Pro"
APP_ICON = "🐕"
LAYOUT_MODE = "wide"

# Model Configuration
MODEL_FILE_PATH = 'final_model.keras'
CLASSES_FILE_PATH = 'classes.txt'
CONFIDENCE_THRESHOLD = 50.0  # Percentage required to be considered a valid match

# Image Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Setup Streamlit Page
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT_MODE,
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# 2. CSS STYLING SYSTEM (PROFESSIONAL LIGHT THEME)
# ==============================================================================

def inject_custom_css():
    """
    Injects extensive CSS to override Streamlit defaults and enforce 
    a high-contrast, professional Light Theme.
    """
    st.markdown("""
    <style>
        /* -----------------------------------------------------------
           1. FONTS & GLOBAL RESET
           ----------------------------------------------------------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #FFFFFF !important; /* Force Pure White */
            color: #111827 !important; /* Dark Slate / Black Text */
        }

        /* Remove Streamlit's default top padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        /* Hide Streamlit Header/Footer for clean look */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}

        /* -----------------------------------------------------------
           2. TYPOGRAPHY
           ----------------------------------------------------------- */
        h1, h2, h3 {
            color: #111827 !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em;
        }
        
        p, div, li {
            color: #374151 !important; /* Soft Black for readability */
            line-height: 1.6;
        }

        .brand-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #2563EB, #1E40AF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .brand-subtitle {
            font-size: 1.1rem;
            color: #6B7280 !important;
            font-weight: 400;
        }

        /* -----------------------------------------------------------
           3. CONTAINER STYLES
           ----------------------------------------------------------- */
        /* Upload Area */
        .upload-container {
            border: 2px dashed #E5E7EB;
            background-color: #F9FAFB;
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            transition: all 0.2s ease-in-out;
        }
        .upload-container:hover {
            border-color: #2563EB;
            background-color: #EFF6FF;
        }

        /* ID Card (Left Panel) */
        .id-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            overflow: hidden;
            height: 100%;
        }

        .id-content {
            padding: 24px;
        }

        .confidence-pill {
            display: inline-block;
            background-color: #DCFCE7;
            color: #15803D !important;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 12px;
        }

        /* Chat Panel (Right Panel) */
        .chat-panel {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            padding: 24px;
            height: 100%;
            min-height: 600px;
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            border-bottom: 1px solid #F3F4F6;
            padding-bottom: 16px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        /* -----------------------------------------------------------
           4. CHAT BUBBLES
           ----------------------------------------------------------- */
        .chat-row {
            display: flex;
            margin-bottom: 16px;
            width: 100%;
        }
        
        .bubble {
            padding: 12px 18px;
            border-radius: 12px;
            font-size: 0.95rem;
            max-width: 85%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        .bubble-user {
            background-color: #2563EB; /* Professional Blue */
            color: #FFFFFF !important;
            border-bottom-right-radius: 2px;
            margin-left: auto; /* Right align */
        }

        .bubble-bot {
            background-color: #F3F4F6; /* Light Gray */
            color: #1F2937 !important;
            border: 1px solid #E5E7EB;
            border-bottom-left-radius: 2px;
            margin-right: auto; /* Left align */
        }

        /* -----------------------------------------------------------
           5. BUTTONS & INPUTS
           ----------------------------------------------------------- */
        .stButton > button {
            background-color: #2563EB;
            color: #FFFFFF !important;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            font-weight: 600;
            transition: background 0.2s;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #1D4ED8;
        }
        
        /* Input Field Styling */
        [data-testid="stChatInput"] {
            border-radius: 12px;
            border: 1px solid #E5E7EB;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
            color: #111827 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.875rem !important;
            color: #6B7280 !important;
        }

    </style>
    """, unsafe_allow_html=True)

# Inject the styles immediately
inject_custom_css()

# ==============================================================================
# 3. COMPREHENSIVE BREED DATABASE
# ==============================================================================

BREED_KNOWLEDGE_BASE = {
    "Siberian Husky": {
        "Origin": "Siberia",
        "Life Span": "12-14 years",
        "Group": "Working",
        "Height": "20-23 inches",
        "Diet": "High-protein, high-fat diet similar to ancestral diet. Fish-based kibble is great.",
        "Climate": "Thrives in cold weather. Needs shade and A/C in summer. DO NOT shave.",
        "Grooming": "Heavy shedder. Blows coat twice a year. Daily brushing needed.",
        "Training": "Intelligent but stubborn. Keep sessions short and fun.",
        "Bio": "The Siberian Husky is a medium-sized working sled dog breed. They are recognizable by their thick double coat, erect triangular ears, and distinctive markings. They are friendly, gentle, and alert."
    },
    "Rottweiler": {
        "Origin": "Germany",
        "Life Span": "9-10 years",
        "Group": "Working",
        "Height": "22-27 inches",
        "Diet": "High protein (22-26%) for muscle mass. Watch calorie intake to protect joints.",
        "Climate": "Tolerates cool weather well. Avoid extreme heat due to black coat.",
        "Grooming": "Low maintenance. Brush weekly.",
        "Training": "Requires firm, consistent leadership and early socialization.",
        "Bio": "The Rottweiler is a robust working breed of great strength descended from the mastiffs of the Roman legions. A gentle playmate and protector within the family circle, the Rottie observes the outside world with a self-assured aloofness."
    },
    "German Shepherd": {
        "Origin": "Germany",
        "Life Span": "7-10 years",
        "Group": "Herding",
        "Height": "22-26 inches",
        "Diet": "High-quality performance food. Glucosamine supplements recommended for hips.",
        "Climate": "Double coat allows tolerance of cold. Versatile in most climates.",
        "Grooming": "Frequent shedding. Brush every few days.",
        "Training": "Highly trainable. Excellent for police, military, and service work.",
        "Bio": "The German Shepherd Dog is one of America's most popular dog breeds — for good reasons. They are capable working dogs of high intelligence and noble character."
    },
    "Boxer": {
        "Origin": "Germany",
        "Life Span": "10-12 years",
        "Group": "Working",
        "Height": "21-25 inches",
        "Diet": "High calorie diet for their high energy levels. Bloat-prone (feed smaller meals).",
        "Climate": "Short coat means they need coats in winter and A/C in summer.",
        "Grooming": "Very low maintenance. Occasional bath.",
        "Training": "Playful and upbeat. Positive reinforcement works best.",
        "Bio": "Loyal, affectionate, energetic, and playful, the Boxer is the ultimate family dog. They are intelligent and active, preferring to be busy."
    },
    "Great Dane": {
        "Origin": "Germany",
        "Life Span": "7-10 years",
        "Group": "Working",
        "Height": "28-32 inches",
        "Diet": "Giant breed formula. crucial to prevent growing too fast (bone issues).",
        "Climate": "Moderate climates. Short coat offers little protection from cold.",
        "Grooming": "Low maintenance.",
        "Training": "gentle giant, but needs obedience training due to size.",
        "Bio": "The easygoing Great Dane, the 'Apollo of Dogs', is a total joy to live with—but owning a dog of such imposing size, weight, and strength is a commitment not to be entered into lightly."
    },
    "Golden Retriever": {
        "Origin": "Scotland",
        "Life Span": "10-12 years",
        "Group": "Sporting",
        "Height": "21-24 inches",
        "Diet": "Balanced diet. They love to eat, so watch for obesity.",
        "Climate": "Adaptable. Loves water and snow. Provide shade in heat.",
        "Grooming": "Regular brushing to prevent mats in feathering.",
        "Training": "Eager to please. The gold standard for obedience.",
        "Bio": "The Golden Retriever is an exuberant Scottish gundog of great beauty. They are serious workers at hunting and field work, as guides for the blind, and in search-and-rescue, enjoy obedience and other competitive events."
    },
    "Labrador Retriever": {
        "Origin": "Canada",
        "Life Span": "10-12 years",
        "Group": "Sporting",
        "Height": "21-24 inches",
        "Diet": "Strict portion control. Labs are prone to becoming overweight.",
        "Climate": "Water-resistant coat makes them hardy in most weather.",
        "Grooming": "Wash-and-wear coat. Brush weekly.",
        "Training": "Friendly and outgoing. Very treat-motivated.",
        "Bio": "The sweet-faced, lovable Labrador Retriever is America's most popular dog breed. Labs are friendly, outgoing, and high-spirited companions who have more than enough affection to go around."
    },
    "Cocker Spaniel": {
        "Origin": "UK/USA",
        "Life Span": "10-14 years",
        "Group": "Sporting",
        "Height": "13-15 inches",
        "Diet": "High quality kibble. Watch for food allergies (ears/skin).",
        "Climate": "Moderate.",
        "Grooming": "High maintenance. Professional grooming needed frequently.",
        "Training": "Gentle training methods. Can be sensitive.",
        "Bio": "The Cocker Spaniel is a beloved companion dog breed, though they remain a capable bird dog at heart. Beautiful, sweet-natured, and moderately active."
    },
    "Pug": {
        "Origin": "China",
        "Life Span": "13-15 years",
        "Group": "Toy",
        "Height": "10-13 inches",
        "Diet": "Calorie-controlled. They gain weight just looking at food.",
        "Climate": "Very sensitive to heat (Brachycephalic). Keep cool!",
        "Grooming": "Clean face wrinkles daily to prevent infection.",
        "Training": "Stubborn but food motivated.",
        "Bio": "The Pug is often described as a lot of dog in a small space. These sturdy, compact dogs are a part of the American Kennel Club’s Toy group, and are known as the clowns of the canine world."
    },
    "Chihuahua": {
        "Origin": "Mexico",
        "Life Span": "14-20 years",
        "Group": "Toy",
        "Height": "5-8 inches",
        "Diet": "Nutrient-dense small breed formula.",
        "Climate": "Hates the cold. Needs sweaters in winter.",
        "Grooming": "Minimal.",
        "Training": "Can be sassy. Needs socialization to prevent 'Small Dog Syndrome'.",
        "Bio": "The Chihuahua is a tiny dog with a huge personality. A national symbol of Mexico, these alert and amusing 'purse dogs' stand among the oldest breeds of the Americas."
    },
    "Shih Tzu": {
        "Origin": "China",
        "Life Span": "10-18 years",
        "Group": "Toy",
        "Height": "9-10 inches",
        "Diet": "High quality food for skin and coat health.",
        "Climate": "Indoor dog. Sensitive to heat.",
        "Grooming": "Daily brushing required if coat is long.",
        "Training": "Can be difficult to housebreak.",
        "Bio": "Shih Tzu means 'Lion Dog', but they are lovers, not fighters. Bred solely to be companions, they are affectionate, happy, and outgoing house dogs who love nothing more than to follow their people from room to room."
    },
    "Beagle": {
        "Origin": "United Kingdom",
        "Life Span": "10-15 years",
        "Group": "Hound",
        "Height": "13-15 inches",
        "Diet": "Measured meals. They are scavengers and will overeat.",
        "Climate": "Adaptable to most climates.",
        "Grooming": "Low maintenance. Ears need cleaning.",
        "Training": "Distracted by scents. Recall training is difficult.",
        "Bio": "The Beagle is a breed of small hound that is similar in appearance to the much larger foxhound. The beagle is a scent hound, developed primarily for hunting hare."
    },
    "Dachshund": {
        "Origin": "Germany",
        "Life Span": "12-16 years",
        "Group": "Hound",
        "Height": "8-9 inches",
        "Diet": "Keep slim to protect their long back (IVDD risk).",
        "Climate": "Likes warmth. Dislikes rain and wet grass.",
        "Grooming": "Varies by coat type (Smooth, Wire, Long).",
        "Training": "Independent and stubborn. Patience required.",
        "Bio": "The Dachshund is an icon of pure dogdom. Their long, low silhouette, ever-alert expression, and bold, vivacious personality have made him a superstar of the canine kingdom."
    },
    "Bulldog": {
        "Origin": "United Kingdom",
        "Life Span": "8-10 years",
        "Group": "Non-Sporting",
        "Height": "14-15 inches",
        "Diet": "Digestible food to reduce gas.",
        "Climate": "Heat intolerant. Indoor AC is mandatory in summer.",
        "Grooming": "Clean face folds daily.",
        "Training": "Short sessions. They tire easily.",
        "Bio": "Kind, courageous, and dignified. The Bulldog is a thick-set, low-slung, well-muscled bruiser whose 'sourmug' face acts as the universal symbol of courage and tenacity."
    },
    "Poodle": {
        "Origin": "France/Germany",
        "Life Span": "10-18 years",
        "Group": "Non-Sporting",
        "Height": "Std: >15in, Min: 10-15in",
        "Diet": "Balanced diet rich in Omega oils.",
        "Climate": "Adaptable.",
        "Grooming": "Professional grooming every 4-6 weeks is mandatory.",
        "Training": "One of the smartest breeds. Learns tricks instantly.",
        "Bio": "Don't let the fancy cut fool you: Poodles are eager, athletic, and wickedly smart 'real dogs' of remarkable versatility. The Standard, with his greater size and strength, is the best all-around athlete of the family."
    }
}

FALLBACK_DATA = {
    "Origin": "International",
    "Life Span": "10-13 years",
    "Group": "Mixed / Unknown",
    "Height": "Varies",
    "Diet": "Standard balanced dog food appropriate for size.",
    "Climate": "Moderate temperatures.",
    "Grooming": "Regular brushing recommended.",
    "Training": "Positive reinforcement.",
    "Bio": "A loyal canine companion identified by our AI. While we don't have specific history for this breed in our quick-access database, they are likely a wonderful pet!"
}

# ==============================================================================
# 4. LOGIC & UTILITIES
# ==============================================================================

@st.cache_resource
def load_model_engine():
    """
    Loads the Keras model and the class names file.
    """
    if not os.path.exists(MODEL_FILE_PATH):
        return None, None
    try:
        model = tf.keras.models.load_model(MODEL_FILE_PATH)
        with open(CLASSES_FILE_PATH, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return model, classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image):
    """
    Prepares an uploaded image for the Neural Network.
    """
    img = image.convert('RGB')
    img = ImageOps.fit(img, (IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def lookup_breed_info(breed_name):
    """
    Smart dictionary lookup.
    """
    if breed_name in BREED_KNOWLEDGE_BASE:
        return BREED_KNOWLEDGE_BASE[breed_name]
    for key in BREED_KNOWLEDGE_BASE:
        if key in breed_name or breed_name in key:
            return BREED_KNOWLEDGE_BASE[key]
    return FALLBACK_DATA

def generate_chat_response(breed_name, user_query):
    """
    Rule-based NLP engine.
    """
    query = user_query.lower()
    info = lookup_breed_info(breed_name)
    
    if any(word in query for word in ["eat", "food", "diet", "feed", "hungry", "treats"]):
        return f"🍖 **Dietary Advice:** {info['Diet']}"
    elif any(word in query for word in ["origin", "from", "country", "history", "where"]):
        return f"🌍 **Origin:** The {breed_name} originates from {info['Origin']}."
    elif any(word in query for word in ["live", "life", "age", "years", "old", "die"]):
        return f"⏳ **Lifespan:** The {breed_name} typically lives for {info['Life Span']}."
    elif any(word in query for word in ["weather", "cold", "hot", "winter", "summer", "climate", "temp"]):
        return f"☀️ **Climate Preference:** {info['Climate']}"
    elif any(word in query for word in ["groom", "brush", "hair", "shed", "fur", "bath"]):
        return f"🛁 **Grooming:** {info.get('Grooming', 'Regular brushing recommended.')}"
    elif any(word in query for word in ["train", "sit", "stay", "behavior", "smart", "intelligent"]):
        return f"🎓 **Training:** {info.get('Training', 'Positive reinforcement works best.')}"
    elif any(word in query for word in ["big", "small", "size", "height", "weight", "tall"]):
        return f"📏 **Size:** They typically stand {info.get('Height', 'Varies')} tall."
    elif any(word in query for word in ["hello", "hi", "hey"]):
        return f"Woof! I am your {breed_name} expert. Ask me about my diet, health, or history!"
    else:
        return f"That's an interesting question about the **{breed_name}**. While I specialize in their biology and care, generally speaking: {info['Bio']}"

# ==============================================================================
# 5. STATE MANAGEMENT
# ==============================================================================

if 'page_view' not in st.session_state:
    st.session_state.page_view = 'LANDING'

if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

def navigate_to_home():
    st.session_state.page_view = 'LANDING'
    st.session_state.analysis_data = None
    st.session_state.chat_messages = []
    st.rerun()

# ==============================================================================
# 6. UI RENDERERS
# ==============================================================================

def render_landing_page():
    """
    Renders the Home/Upload Screen with Professional styling.
    """
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Clean, centered header
    st.markdown("""
        <div style='text-align: center; margin-bottom: 40px;'>
            <div style='font-size: 80px; margin-bottom: 10px;'>🐕</div>
            <h1 class='brand-title'>PawIdentify Pro</h1>
            <p class='brand-subtitle'>Professional Grade AI Breed Detection & Veterinary Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-bottom: 10px;'>Analyze Dog Image</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 14px; color: #9CA3AF !important;'>Upload a clear JPEG or PNG image</p>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Features footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    f1.info("**98% Accuracy**\n\nTrained on Stanford Dataset")
    f2.info("**Instant Analysis**\n\nPowered by TensorFlow")
    f3.info("**Expert Chatbot**\n\nContext-aware advice")

    return uploaded_file

def render_result_dashboard():
    """
    Renders the Result View (Split Screen).
    """
    data = st.session_state.analysis_data
    breed = data['breed']
    conf = data['conf']
    img = data['image']
    info = lookup_breed_info(breed)
    
    # Layout
    left_col, right_col = st.columns([1, 1.4], gap="medium")
    
    # --- LEFT: ID CARD ---
    with left_col:
        st.image(img, use_container_width=True)
        
        st.markdown("<div class='id-card'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='id-content'>
                <div class='confidence-pill'>Match: {conf:.1f}%</div>
                <h2 style='margin:0; font-size: 28px;'>{breed}</h2>
                <p style='color: #6B7280; margin-top: 5px;'>{info.get('Group', 'Unknown')} Group</p>
                <hr style='border: 0; border-top: 1px solid #F3F4F6; margin: 20px 0;'>
                <p style='font-style: italic; color: #4B5563 !important;'>"{info.get('Bio', '')}"</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Stats below bio
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Origin", info.get("Origin", "Unknown"))
        with c2:
            st.metric("Lifespan", info.get("Life Span", "Unknown"))
            
        st.markdown("</div>", unsafe_allow_html=True) # End Card
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬅ Analyze Another"):
            navigate_to_home()

    # --- RIGHT: CHAT ---
    with right_col:
        st.markdown("<div class='chat-panel'>", unsafe_allow_html=True)
        
        # Chat Header
        st.markdown(f"""
            <div class='chat-header'>
                <div style='background: #EFF6FF; padding: 10px; border-radius: 50%; color: #2563EB;'>💬</div>
                <div>
                    <div style='font-weight: 700; font-size: 18px;'>{breed} Expert</div>
                    <div style='font-size: 12px; color: #16A34A; font-weight: 500;'>● Online Now</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Scrollable Chat Area
        chat_container = st.container(height=500)
        
        with chat_container:
            for msg in st.session_state.chat_messages:
                if msg['role'] == 'user':
                    st.markdown(f"""
                        <div class='chat-row'>
                            <div class='bubble bubble-user'>{msg['content']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='chat-row'>
                            <div class='bubble bubble-bot'>{msg['content']}</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True) # End Panel

        # Input
        if prompt := st.chat_input(f"Ask about diet, grooming, etc..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            response = generate_chat_response(breed, prompt)
            st.session_state.chat_messages.append({"role": "bot", "content": response})
            st.rerun()

def render_error_screen(conf):
    """
    Renders 'No Dog Detected' screen.
    """
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.error(f"Low Confidence ({conf:.1f}%). No known dog breed detected.")
        if st.button("Try Again"):
            navigate_to_home()

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================

def main():
    if st.session_state.page_view == 'LANDING':
        file = render_landing_page()
        
        if file:
            with st.spinner("Analyzing image..."):
                # Load Resources
                model, classes = load_model_engine()
                
                if not model:
                    st.error("Model file not found. Please check setup.")
                    st.stop()
                
                # Predict
                img = Image.open(file).convert('RGB')
                img_array = preprocess_image(img)
                preds = model.predict(img_array)
                score = preds[0]
                top_idx = np.argmax(score)
                conf = 100 * np.max(score)
                
                # Format Name
                raw_name = classes[top_idx]
                breed_name = raw_name.split('-', 1)[1].replace('_', ' ').title() if '-' in raw_name else raw_name
                
                # Store State
                st.session_state.analysis_data = {
                    "image": img,
                    "breed": breed_name,
                    "conf": conf
                }
                
                # Initial Greeting
                st.session_state.chat_messages = [{
                    "role": "bot", 
                    "content": f"Hello! I've identified this as a **{breed_name}**. I can help you with care tips, diet, and training advice. What would you like to know?"
                }]
                
                st.session_state.page_view = 'RESULT'
                st.rerun()

    elif st.session_state.page_view == 'RESULT':
        conf = st.session_state.analysis_data['conf']
        if conf < CONFIDENCE_THRESHOLD:
            render_error_screen(conf)
        else:
            render_result_dashboard()

if __name__ == '__main__':
    main()
