import streamlit as st
import pandas as pd
import os
import cv2
from fer import FER
from collections import deque, Counter
import time
import numpy as np
from googleapiclient.discovery import build
import random
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# --- File Config ---
CSV_FILE = "users.csv"

GENRE_OPTIONS = [
    "Bollywood","Heartbreak","Retro / Old Classics","Punjabi",
    "Bhojpuri","Marathi","Pop","K-Pop","Hip-Hop / Rap","Acoustic / Soft Pop","Vibe / Aesthetic Music",
    "TikTok / Instagram Trends","Rap Battles / Freestyle",
    "Motivational / Gym","Party","Study / Focus","Road Trip / Travel","Rainy Day Songs","Sleep / Calm","Morning Vibes",
    "Late Night Vibes","Dance / Workout Mix","Romantic Chill","Healing",
    "Sufi / Qawwali","Spiritual / Devotional","Instrumental",
]

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# --- Smart Emotion-Keyword Mapping ---
EMOTION_KEYWORDS = {
    'happy': ['upbeat', 'joyful', 'dance', 'celebratory', 'positive', 'energetic'],
    'sad': ['sad', 'melancholic', 'emotional', 'heartbreak', 'soothing', 'calm'],
    'angry': ['intense', 'powerful', 'rock', 'aggressive', 'cathartic', 'strong', 'motivational'],
    'surprise': ['surprise', 'unexpected', 'epic', 'dramatic', 'exciting', 'orchestral'],
    'fear': ['calm', 'soothing', 'peaceful', 'meditation', 'ambient', 'comforting'],
    'disgust': ['empowering', 'strong', 'confident', 'anthem', 'defiant'],
    'neutral': ['chill', 'background', 'focus', 'relaxing'],
}

# ---Fetch Songs from YouTube  ---
def youtube_search(query, max_results=5):
    try:
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoCategoryId="10",
            videoEmbeddable="true",
            safeSearch="strict",
            maxResults=max_results * 3, 
            order="relevance" 
        )
        response = request.execute()
        return response.get("items", [])
    except Exception as e:
        st.error(f"YouTube API error for query '{query}': {e}")
        return []

def get_songs(emotion, genres=None, language=None, max_results=5, search_offset=0):
    emotion_words = EMOTION_KEYWORDS.get(emotion, [emotion])
    
    genre_str = " ".join(genres) if genres else ""
    
    queries = []
    start_index = search_offset % len(emotion_words)
    emotion_subset = emotion_words[start_index:start_index+2] + emotion_words[:1]

    for emotion_word in emotion_subset:
        base_query = f"{emotion_word} {genre_str}".strip()

        if language and language.lower() != "both":
            query = f"{base_query} {language} music"
        else:
            query = f"{base_query} music"

        variation_terms = ["songs", "playlist", "mix", "hits", "tracks"]
        variation = variation_terms[search_offset % len(variation_terms)]
        query = f"{query} {variation}"

        queries.append(query)

    queries = list(set(queries)) 

   
    all_results = []
    seen_video_ids = set()

    for query in queries:
        items = youtube_search(query, max_results)

        for item in items:
            video_id = item["id"]["videoId"]
            if video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)

            title = item["snippet"]["title"]
            artist = item["snippet"]["channelTitle"]
            url = f"https://www.youtube.com/watch?v={video_id}"

            all_results.append((title, url, query, artist, video_id))

            if len(all_results) >= max_results * 2:
                break

        if len(all_results) >= max_results * 2:
            break

    return all_results[:max_results * 2]


# --- Functions to load/save users ---
def load_users():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        return pd.DataFrame(columns=["Name", "Age", "Gender", "Music_Language", "Genres"])

def save_users(df):
    df.to_csv(CSV_FILE, index=False)

# Create tabs
tab1, tab2 = st.tabs(["🎵 Main App", "📊 Analysis Dashboard"])

with tab1:
        # --- Streamlit App ---
    st.set_page_config(page_title="Emotion-Based Music Recommender", layout="centered")
    st.title("🎵 Emotion-Based Music Recommendation System")

    users_df = load_users()

# --- 1. PROFILE FORM ---
    with st.expander("👤 Fill Your Profile", expanded=True):
        if 'user_info' not in st.session_state:
            st.session_state['user_info'] = {}

        name = st.text_input("Name", value=st.session_state['user_info'].get("Name", ""))

       
        if name and not all(char.isalpha() for char in name):
            st.error("Please enter a valid name (letters only).")
            st.stop()

        age = st.number_input("Age", min_value=1, max_value=100, step=1,
                            value=st.session_state['user_info'].get("Age"))
        gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                            index=["Male", "Female", "Other"].index(st.session_state['user_info'].get("Gender", "Male")))
        music_language = st.selectbox("Preferred Music Language", ["Hindi", "English", "Both"],
                                    index=["Hindi", "English", "Both"].index(st.session_state['user_info'].get("Music_Language", "Hindi")))
        
        default_genres = st.session_state['user_info'].get("Genres", ["Pop"])
        if isinstance(default_genres, str):
            default_genres = [genre.strip() for genre in default_genres.split(",")]
        
        genres = st.multiselect("Preferred Genres", GENRE_OPTIONS, default=default_genres)

        if st.button("Save Profile"):
            if not name or not genres:
                st.warning("Please fill all fields and choose at least one genre.")
            else:
                # Check if same user exists
                exists = ((users_df['Name'] == name) & (users_df['Age'] == age)).any() if not users_df.empty else False
                new_row = {"Name": name, "Age": age, "Gender": gender, 
                        "Music_Language": music_language, "Genres": ", ".join(genres)}

                if exists:
                    # Update existing
                    users_df.loc[(users_df['Name'] == name) & (users_df['Age'] == age),
                                ['Gender', 'Music_Language', 'Genres']] = [gender, music_language, ", ".join(genres)]
                    save_users(users_df)
                    st.success(f"Welcome back, {name}! Your preferences updated.")
                    user_info = users_df[(users_df['Name'] == name) & (users_df['Age'] == age)].iloc[0].to_dict()
                else:
                    users_df = pd.concat([users_df, pd.DataFrame([new_row])], ignore_index=True)
                    save_users(users_df)
                    st.success(f"Welcome {name}! Your profile saved.")
                    user_info = new_row

                st.session_state['user_info'] = user_info
                st.session_state['profile_saved'] = True

    # Stop if profile not filled
    if 'user_info' not in st.session_state or not st.session_state['user_info']:
        st.info("Please fill your profile above to start emotion detection.")
        st.stop()

    user_info = st.session_state['user_info']

    # --- 2. EMOTION DETECTION ---

    st.subheader(f"🎥 Live Emotion Detection for {user_info['Name']}")

    if 'start_emotion' not in st.session_state:
        st.session_state['start_emotion'] = False
    if 'current_emotion' not in st.session_state:
        st.session_state['current_emotion'] = "Detecting..."
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    if 'recommend_ready' not in st.session_state:
        st.session_state['recommend_ready'] = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Webcam"):
            st.session_state['start_emotion'] = True
            st.session_state['recommend_ready'] = False
    with col2:
        if st.button("Stop Webcam"):
            st.session_state['start_emotion'] = False
            st.session_state['recommend_ready'] = True 

    frame_placeholder = st.empty()
    emotion_placeholder = st.empty()

    @st.cache_resource
    def load_detector():
        detector = FER(mtcnn=False)
        time.sleep(0.5)  
        return detector

    detector = load_detector()
    emotion_history = deque(maxlen=15)
    box_history = deque(maxlen=8)
    CONFIDENCE_THRESHOLD = 0.4

    if st.session_state['start_emotion']:
        cap = cv2.VideoCapture(0)
        frame_count = 0

        while st.session_state['start_emotion'] and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame.")
                break

            small = cv2.resize(frame, (320, 240))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            if frame_count % 2 == 0:
                results = detector.detect_emotions(rgb)
                if results:
                    top = results[0]
                    (x, y, w, h) = top["box"]
                    emotions = top["emotions"]
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[dominant_emotion]

                    if confidence > CONFIDENCE_THRESHOLD:
                        emotion_history.append(dominant_emotion)
                    else:
                        emotion_history.append("Uncertain")

                    scale_x = frame.shape[1] / small.shape[1]
                    scale_y = frame.shape[0] / small.shape[0]
                    x1, y1, w1, h1 = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)
                    box_history.append((x1, y1, w1, h1))
                    last_conf = confidence
                else:
                    emotion_history.append("No face")
                    last_conf = 0

            filtered = [e for e in emotion_history if e not in ("Uncertain", "No face")]
            if filtered:
                smooth_emotion = Counter(filtered).most_common(1)[0][0]
            else:
                smooth_emotion = st.session_state.get('last_valid_emotion', "Detecting...")

            st.session_state['current_emotion'] = smooth_emotion
            st.session_state['last_valid_emotion'] = smooth_emotion
            st.session_state.emotion_history.append(smooth_emotion)

            if box_history:
                avg_box = np.mean(box_history, axis=0).astype(int)
                x1, y1, w1, h1 = avg_box
                cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
                if smooth_emotion not in ("Detecting...", "No face", "Uncertain"):
                    label = f"{smooth_emotion} ({last_conf*100:.1f}%)"
                else:
                    label = smooth_emotion
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            emotion_placeholder.markdown(f"### Detected Emotion: **{smooth_emotion}**")

            frame_count += 1
            time.sleep(0.02)

        cap.release()
    else:
        frame_placeholder.info("Webcam is off")
        emotion_placeholder.markdown(f"### Last Detected Emotion: **{st.session_state['current_emotion']}**")


    # --- 3. RECOMMENDATION SECTION ---
    if st.session_state['recommend_ready']:
        st.subheader("🎶 Music Recommendations")
        
        if 'search_count' not in st.session_state:
            st.session_state.search_count = 0
        
        if st.button("🎵 Recommend Songs"):
            emotion = st.session_state['current_emotion']
            genres = user_info["Genres"].split(", ")
            language = user_info["Music_Language"]
            
            st.markdown("---")

            
            with st.spinner(f'🔍 Finding fresh {emotion} songs...'):
                songs = get_songs(emotion, genres=genres, language=language, 
                                max_results=5, search_offset=st.session_state.search_count)
            
            st.session_state.search_count += 1
            
            if not songs:
                st.warning("No songs found. Try changing filters.")
            else:
                random.shuffle(songs)
                display_songs = songs[:5]  
                
                for idx, (title, url, query, artist, video_id) in enumerate(display_songs):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**🎵 {title}**")
                            st.markdown(f"*Artist: {artist}*")
                            st.markdown(f"**🔍 Smart Search:** `{query}`")
                        with col2:
                            st.caption(f"ID: {video_id[:8]}...")
                            st.caption(f"Result: {idx+1}")
                        
                        st.video(url, format="video/mp4", start_time=0)
                        st.markdown("---")
                
                
                st.caption("💡 Click 'Recommend Songs' again for different results!")


with tab2:
    st.title("📊 Analysis Dashboard")
    st.subheader("User & Emotion Analytics")
    st.markdown("---")

    # Safe initialization
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = deque(maxlen=15)
    
    # Convert deque to list for analysis
    emotion_list = list(st.session_state.emotion_history)
    
    # Section 1: User Statistics
    
    if not users_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Total Users", len(users_df))
        with col2:
            avg_age = int(users_df['Age'].mean()) if not users_df.empty else 0
            st.metric("🎂 Average Age", avg_age)
        with col3:
            try:
                all_genres_list = users_df['Genres'].str.split(', ').explode()
                popular_genre = all_genres_list.mode()
                popular_genre = popular_genre[0] if not popular_genre.empty else "No data"
            except:
                popular_genre = "No data"
            st.metric("🎵 Popular Genre", popular_genre)
    else:
        st.info("👤 No user data available. Save a profile first to see analytics.")
    
    st.markdown("---")
    
    
    # Section 3: User Demographics (only if data exists)
    if not users_df.empty:
        st.subheader("👥 User Demographics")
        
        # Age Distribution
        st.write("**Age Distribution**")
        age_chart = users_df['Age'].value_counts().sort_index()
        st.bar_chart(age_chart)
        
        st.markdown("---")
        
        # Gender Distribution
        st.write("**Gender Distribution**")
        gender_counts = users_df['Gender'].value_counts()
        
        if not gender_counts.empty:
            fig = px.pie(
                values=gender_counts.values, 
                names=gender_counts.index, 
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Section 4: Music Preferences (only if data exists)
    if not users_df.empty:
        st.markdown("---")
        st.subheader("🎵 Music Preferences")
        
        st.write("**Genre Popularity**")
        all_genres = users_df['Genres'].str.split(', ').explode()
        genre_counts = all_genres.value_counts()
        st.bar_chart(genre_counts.head(8))
        
        st.markdown("---")
        
        st.write("**Language Preference**")
        lang_counts = users_df['Music_Language'].value_counts()
            
        if not lang_counts.empty:
                lang_df = pd.DataFrame({
                    'Language': lang_counts.index,
                    'Users': lang_counts.values
                })
                st.dataframe(lang_df, hide_index=True, use_container_width=True)
    
    # Simple refresh button
    st.markdown("---")
    if st.button("🔄 Refresh Analytics", type="secondary"):
        st.rerun()


