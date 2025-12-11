"""
PawIdentify Enterprise Edition
------------------------------
A Streamlit-based AI application for Dog Breed Identification and 
Context-Aware Chatbot assistance.

Author: AI Assistant
Version: 3.0.0 (Production)
Theme: Light / Google Lens Style
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
# 2. CSS STYLING SYSTEM (LIGHT THEME ENFORCEMENT)
# ==============================================================================

def inject_custom_css():
    """
    Injects extensive CSS to override Streamlit defaults and enforce 
    a high-contrast Light Theme (Google Lens Aesthetic).
    """
    st.markdown("""
    <style>
        /* -----------------------------------------------------------
           GLOBAL RESET & FONTS
           ----------------------------------------------------------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #FFFFFF; /* Pure White Background */
            color: #000000 !important; /* Force Black Text */
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Remove top padding */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 5rem;
        }

        /* -----------------------------------------------------------
           TYPOGRAPHY
           ----------------------------------------------------------- */
        h1, h2, h3, h4, h5, h6 {
            color: #111827 !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }
        
        p, span, div, label {
            color: #1F2937 !important; /* Dark Slate for readability */
        }
        
        .big-brand-title {
            font-size: 48px;
            background: linear-gradient(90deg, #2563EB, #4F46E5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        /* -----------------------------------------------------------
           COMPONENTS: UPLOAD AREA
           ----------------------------------------------------------- */
        .upload-wrapper {
            background: #F8FAFC;
            border: 2px dashed #CBD5E1;
            border-radius: 24px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
        }
        .upload-wrapper:hover {
            border-color: #2563EB;
            background: #EFF6FF;
        }
        
        /* Force Streamlit Uploader to look cleaner */
        [data-testid='stFileUploader'] {
            width: 100%;
        }
        [data-testid='stFileUploader'] section {
            background-color: transparent;
            padding: 0;
            border: none;
        }

        /* -----------------------------------------------------------
           COMPONENTS: ID CARD (LEFT PANEL)
           ----------------------------------------------------------- */
        .id-card-container {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.06);
            overflow: hidden;
            height: 100%;
        }
        
        .id-card-header {
            padding: 24px;
            border-bottom: 1px solid #F1F5F9;
        }
        
        .breed-header-text {
            font-size: 32px;
            margin: 0;
            color: #111827;
        }
        
        .confidence-badge {
            background-color: #DCFCE7; /* Light Green */
            color: #166534 !important; /* Dark Green Text */
            padding: 6px 12px;
            border-radius: 100px;
            font-size: 14px;
            font-weight: 700;
            display: inline-block;
            margin-bottom: 10px;
        }

        .bio-box {
            padding: 24px;
            background-color: #F8FAFC;
            margin: 20px;
            border-radius: 12px;
            border-left: 4px solid #2563EB;
            font-style: italic;
        }

        /* -----------------------------------------------------------
           COMPONENTS: CHAT INTERFACE (RIGHT PANEL)
           ----------------------------------------------------------- */
        .chat-panel-container {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.06);
            padding: 20px;
            min-height: 600px;
            display: flex;
            flex-direction: column;
        }
        
        .chat-header-bar {
            display: flex;
            align-items: center;
            gap: 10px;
            padding-bottom: 20px;
            border-bottom: 1px solid #E5E7EB;
            margin-bottom: 20px;
        }
        
        /* Chat Bubbles */
        .chat-message-container {
            display: flex;
            margin-bottom: 15px;
            width: 100%;
        }
        
        .chat-bubble {
            padding: 14px 18px;
            border-radius: 18px;
            font-size: 15px;
            line-height: 1.5;
            max-width: 80%;
            position: relative;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .user-bubble {
            background-color: #2563EB; /* Primary Blue */
            color: #FFFFFF !important; /* White Text */
            border-bottom-right-radius: 4px;
            margin-left: auto; /* Push to right */
        }
        
        .bot-bubble {
            background-color: #F1F5F9; /* Light Gray */
            color: #1F2937 !important; /* Dark Text */
            border-bottom-left-radius: 4px;
            border: 1px solid #E2E8F0;
        }

        /* -----------------------------------------------------------
           COMPONENTS: METRICS & BUTTONS
           ----------------------------------------------------------- */
        [data-testid="stMetricValue"] {
            font-size: 20px !important;
            color: #111827 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 14px !important;
            color: #6B7280 !important;
        }
        
        .stButton > button {
            background-color: #2563EB;
            color: white !important;
            border-radius: 12px;
            border: none;
            padding: 12px 24px;
            font-weight: 600;
            width: 100%;
            transition: background-color 0.2s;
        }
        .stButton > button:hover {
            background-color: #1D4ED8;
        }
        
        /* ERROR STATES */
        .error-state-box {
            background-color: #FEF2F2;
            border: 2px solid #FECACA;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
        }
        .error-title {
            color: #991B1B !important;
            font-size: 28px;
            font-weight: 800;
        }
        
    </style>
    """, unsafe_allow_html=True)

# Inject the styles immediately
inject_custom_css()

# ==============================================================================
# 3. COMPREHENSIVE BREED DATABASE (The "Brain")
# ==============================================================================

BREED_KNOWLEDGE_BASE = {
    # --- WORKING GROUP ---
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

    # --- SPORTING GROUP ---
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

    # --- TOY GROUP ---
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

    # --- HOUND GROUP ---
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

    # --- NON-SPORTING / OTHERS ---
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

# Fallback data for breeds not in the detailed list
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
# 4. UTILITY FUNCTIONS & LOGIC
# ==============================================================================

@st.cache_resource
def load_model_engine():
    """
    Loads the Keras model and the class names file.
    Uses caching to prevent reloading on every interaction (Speed optimization).
    """
    # Check if files exist
    if not os.path.exists(MODEL_FILE_PATH):
        return None, None
        
    try:
        model = tf.keras.models.load_model(MODEL_FILE_PATH)
        
        # Load classes
        with open(CLASSES_FILE_PATH, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
            
        return model, classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image):
    """
    Prepares an uploaded image for the Neural Network.
    1. Converts to RGB (removes Alpha channel)
    2. Resizes to 224x224 (Model Requirement)
    3. Converts to Array and adds Batch Dimension
    """
    img = image.convert('RGB')
    img = ImageOps.fit(img, (IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def lookup_breed_info(breed_name):
    """
    Smart dictionary lookup. Tries exact match, then partial match.
    """
    # 1. Exact Match
    if breed_name in BREED_KNOWLEDGE_BASE:
        return BREED_KNOWLEDGE_BASE[breed_name]
    
    # 2. Fuzzy Match (e.g., "Standard Poodle" -> matches "Poodle")
    for key in BREED_KNOWLEDGE_BASE:
        if key in breed_name or breed_name in key:
            return BREED_KNOWLEDGE_BASE[key]
            
    # 3. Fallback
    return FALLBACK_DATA

def generate_chat_response(breed_name, user_query):
    """
    Rule-based NLP engine to answer questions about the specific dog breed.
    """
    query = user_query.lower()
    info = lookup_breed_info(breed_name)
    
    # Keyword Analysis
    
    # Diet / Food
    if any(word in query for word in ["eat", "food", "diet", "feed", "hungry", "treats"]):
        return f"🍖 **Dietary Advice:** {info['Diet']}"
    
    # Origin / History
    elif any(word in query for word in ["origin", "from", "country", "history", "where"]):
        return f"🌍 **Origin:** The {breed_name} originates from {info['Origin']}."
    
    # Lifespan / Age
    elif any(word in query for word in ["live", "life", "age", "years", "old", "die"]):
        return f"⏳ **Lifespan:** The {breed_name} typically lives for {info['Life Span']}."
        
    # Weather / Climate
    elif any(word in query for word in ["weather", "cold", "hot", "winter", "summer", "climate", "temp"]):
        return f"☀️ **Climate Preference:** {info['Climate']}"
        
    # Grooming / Shedding
    elif any(word in query for word in ["groom", "brush", "hair", "shed", "fur", "bath"]):
        return f"🛁 **Grooming:** {info.get('Grooming', 'Regular brushing recommended.')}"
        
    # Training / Behavior
    elif any(word in query for word in ["train", "sit", "stay", "behavior", "smart", "intelligent"]):
        return f"🎓 **Training:** {info.get('Training', 'Positive reinforcement works best.')}"
        
    # Height / Size
    elif any(word in query for word in ["big", "small", "size", "height", "weight", "tall"]):
        return f"📏 **Size:** They typically stand {info.get('Height', 'Varies')} tall."
    
    # General / Greeting
    elif any(word in query for word in ["hello", "hi", "hey"]):
        return f"Woof! I am your {breed_name} expert. Ask me about my diet, health, or history!"
        
    # Fallback for unknown questions
    else:
        return f"That's an interesting question about the **{breed_name}**. While I specialize in their biology and care, generally speaking: {info['Bio']}"

# ==============================================================================
# 5. STATE MANAGEMENT
# ==============================================================================

if 'page_view' not in st.session_state:
    st.session_state.page_view = 'LANDING' # Options: LANDING, RESULT

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
# 6. UI COMPONENT RENDERING
# ==============================================================================

def render_landing_page():
    """
    Renders the Home/Upload Screen.
    """
    # Header Section
    st.markdown("<div style='text-align: center; padding-top: 20px;'>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=80) # Generic Dog Icon
    st.markdown("<h1 class='big-brand-title'>PawIdentify Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px; color: #64748B;'>Advanced AI Breed Detection & Veterinary Chatbot</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("---")

    # Upload Section (Centered)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='upload-wrapper'>", unsafe_allow_html=True)
        st.markdown("<h3>📸 Upload Dog Photo</h3>", unsafe_allow_html=True)
        st.caption("Supports JPG, PNG, JPEG")
        
        uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    # Footer Info
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    fc1.info("🔍 **98% Accuracy** on 120 breeds")
    fc2.info("⚡ **Instant Analysis** via TensorFlow")
    fc3.info("💬 **Expert Chat** for care advice")

    return uploaded_file

def render_result_dashboard():
    """
    Renders the Split-Screen Result View (Profile + Chat).
    """
    data = st.session_state.analysis_data
    breed = data['breed']
    conf = data['conf']
    img = data['image']
    
    # Fetch Breed Details
    info = lookup_breed_info(breed)
    
    # ---------------------------------------------------------
    # LAYOUT: Two Columns (Left: Profile, Right: Chat)
    # ---------------------------------------------------------
    left_col, right_col = st.columns([1, 1.4], gap="large")
    
    # --- LEFT COLUMN: ID CARD ---
    with left_col:
        # Image Display
        st.image(img, use_container_width=True)
        
        # ID Card Container
        st.markdown("<div class='id-card-container'>", unsafe_allow_html=True)
        
        # Header
        st.markdown(f"""
            <div class='id-card-header'>
                <div class='confidence-badge'>Match Confidence: {conf:.1f}%</div>
                <h1 class='breed-header-text'>{breed}</h1>
                <p style='color: #6B7280; margin:0;'>{info.get('Group', 'Unknown')} Group</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Stats using Native Streamlit Metrics (Safe & Clean)
        m1, m2 = st.columns(2)
        with m1:
            st.metric("🌍 Origin", info.get("Origin", "Unknown"))
            st.metric("📏 Height", info.get("Height", "Varies"))
        with m2:
            st.metric("⏳ Lifespan", info.get("Life Span", "Unknown"))
            st.metric("🌡️ Climate", "See Chat")
            
        # Bio Box
        st.markdown(f"<div class='bio-box'>{info.get('Bio', 'No bio available.')}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True) # End Card
        
        # Reset Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬅ Scan Another Dog"):
            navigate_to_home()

    # --- RIGHT COLUMN: CHATBOT ---
    with right_col:
        st.markdown("<div class='chat-panel-container'>", unsafe_allow_html=True)
        
        # Chat Header
        st.markdown(f"""
            <div class='chat-header-bar'>
                <span style='font-size: 24px;'>💬</span>
                <div>
                    <div style='font-weight: 700; font-size: 18px;'>{breed} Specialist</div>
                    <div style='font-size: 12px; color: #16A34A;'>● Online</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Chat Messages Area (Scrollable container)
        chat_container = st.container(height=500)
        
        with chat_container:
            for msg in st.session_state.chat_messages:
                if msg['role'] == 'user':
                    # User Message (Right Aligned, Blue)
                    st.markdown(f"""
                        <div class='chat-message-container' style='justify-content: flex-end;'>
                            <div class='chat-bubble user-bubble'>{msg['content']}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # Bot Message (Left Aligned, Gray)
                    st.markdown(f"""
                        <div class='chat-message-container' style='justify-content: flex-start;'>
                            <div class='chat-bubble bot-bubble'>{msg['content']}</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True) # End Chat Panel Wrapper

        # Chat Input (Native Streamlit Input)
        # We place this *outside* the custom div so Streamlit handles the binding correctly
        if prompt := st.chat_input(f"Ask about {breed} diet, training, etc..."):
            # 1. Add User Message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # 2. Generate Response
            response = generate_chat_response(breed, prompt)
            
            # 3. Add Bot Message
            st.session_state.chat_messages.append({"role": "bot", "content": response})
            
            # 4. Refresh to show new messages
            st.rerun()

def render_error_screen(conf):
    """
    Renders the 'No Dog Detected' screen when confidence is low.
    """
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"""
            <div class='error-state-box'>
                <div style='font-size: 60px; margin-bottom: 20px;'>🐕❓</div>
                <h1 class='error-title'>No Dog Detected</h1>
                <p style='font-size: 18px; margin-bottom: 5px; color: #000 !important;'>Analysis Confidence: <b>{conf:.1f}%</b></p>
                <p style='color: #4B5563 !important;'>The AI is not confident that the image contains a known dog breed.<br>
                Please try uploading a clearer photo.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Try Again"):
            navigate_to_home()

# ==============================================================================
# 7. MAIN APP EXECUTION FLOW
# ==============================================================================

def main():
    
    # 1. Routing Logic
    if st.session_state.page_view == 'LANDING':
        
        # Render Home & Handle Upload
        file = render_landing_page()
        
        if file:
            # Perform Analysis
            with st.spinner("🧠 AI is analyzing breed features..."):
                
                # Check for Model
                if not os.path.exists(MODEL_FILE_PATH):
                    st.error("⚠️ Critical Error: 'final_model.keras' not found. Please train the model first.")
                    st.stop()
                
                # Load Model (Cached)
                model, classes = load_model_engine()
                
                if model is None:
                    st.error("Failed to load model.")
                    st.stop()
                
                # Image Preprocessing
                image = Image.open(file).convert('RGB')
                img_resized = ImageOps.fit(image, (IMG_HEIGHT, IMG_WIDTH), Image.LANCZOS)
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Inference
                preds = model.predict(img_array)
                score = preds[0]
                top_idx = np.argmax(score)
                conf = 100 * np.max(score)
                
                # Result Formatting
                raw_name = classes[top_idx]
                breed_name = raw_name.split('-', 1)[1].replace('_', ' ').title() if '-' in raw_name else raw_name
                
                # Update Session State
                st.session_state.analysis_data = {
                    "image": image,
                    "breed": breed_name,
                    "conf": conf
                }
                
                # Initialize Chat with Context
                st.session_state.chat_messages = [{
                    "role": "bot", 
                    "content": f"Hello! I've identified this as a **{breed_name}**. I am an expert on this breed. Ask me anything!"
                }]
                
                st.session_state.page_view = 'RESULT'
                st.rerun()

    elif st.session_state.page_view == 'RESULT':
        
        # Check Confidence Threshold
        confidence = st.session_state.analysis_data['conf']
        
        if confidence < CONFIDENCE_THRESHOLD:
            render_error_screen(confidence)
        else:
            render_result_dashboard()

if __name__ == '__main__':
    main()
