# 🎵 Emotion-Based Music Recommendation System

A real-time emotion detection system that recommends personalized music based on your facial expressions using AI and webcam technology.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

## 🌟 Features

### 🎭 Real-time Emotion Detection
- Live webcam facial expression analysis using FER (Facial Emotion Recognition)
- Smart emotion smoothing using historical data
- Confidence-based emotion filtering
- Real-time bounding box visualization with MTCNN

### 🎵 Smart Music Recommendations
- Emotion-based song suggestions (Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral)
- Genre and language preferences
- YouTube Data API integration for music videos
- Dynamic search queries for diverse results

### 👤 User Management
- Personalized user profiles with CSV storage
- Age, gender, and music preference tracking
- Persistent user data management

### 📊 Analytics Dashboard
- User demographics visualization
- Music preference analytics using Plotly
- Real-time emotion history tracking
- Interactive charts and graphs

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/emotion-music-system.git
cd emotion-music-system
```
Create virtual environment (optional but recommended)

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Set up environment variables

Create a .env file in the root directory

Add your YouTube Data API v3 key:

env
YOUTUBE_API_KEY=your_youtube_api_key_here
🔑 Getting YouTube API Key
Go to Google Cloud Console

Create a new project or select existing one

Enable YouTube Data API v3

Create credentials (API Key)

Copy the key to your .env file

🎮 Usage
Run the application

bash
streamlit run emotion_music_system.py
Access the web interface

Open your browser to http://localhost:8501

Setup Profile

Fill in your name, age, gender

Select preferred music language and genres from 25+ options

Save your profile

Emotion Detection

Click "Start Webcam" to begin real-time emotion analysis

Allow camera permissions when prompted

View real-time emotion detection with confidence scores

Get Recommendations

Click "Stop Webcam" when ready

Click "Recommend Songs" for personalized music based on detected emotion

Watch embedded YouTube videos directly in the app

📁 Project Structure
text
emotion-music-system/
├── emotion_music_system.py  # Main application
├── requirements.txt         # Dependencies
├── .env                    # Environment variables (create this)
├── users.csv               # User data (auto-generated)
└── README.md              # Project documentation
🛠️ Technologies Used
Frontend: Streamlit

Emotion Detection: FER (Facial Emotion Recognition) + MTCNN

Computer Vision: OpenCV

Machine Learning: TensorFlow, Keras

Data Analysis: Pandas, NumPy

Visualization: Plotly, Matplotlib

Music API: YouTube Data API v3

Data Storage: CSV files

🎯 How It Works
Face Detection: MTCNN detects faces in webcam feed

Emotion Analysis: FER model analyzes facial expressions using deep learning

Emotion Mapping: Converts emotions to music keywords using smart mapping

Smart Search: Generates dynamic YouTube search queries based on emotion and preferences

Personalization: Filters results based on user genre and language preferences

Recommendation: Displays embedded YouTube music videos with metadata

📊 Supported Emotions & Music Mapping
Emotion	Music Keywords	Example Genres
😊 Happy	upbeat, joyful, dance	Pop, Party, Dance, Punjabi
😢 Sad	melancholic, emotional, heartbreak	Acoustic, Romantic Chill, Healing
😠 Angry	intense, powerful, rock	Hip-Hop, Rap, Motivational, Gym
😲 Surprise	epic, dramatic, exciting	Bollywood, Retro, Aesthetic
😨 Fear	calm, soothing, peaceful	Sleep, Calm, Instrumental, Spiritual
🤢 Disgust	empowering, strong, confident	Rap Battles, Freestyle, Anthems
😐 Neutral	chill, background, relaxing	Study, Focus, Ambient, Late Night
🎵 Available Genres (25+ Options)
Regional: Bollywood, Punjabi, Bhojpuri, Marathi

International: Pop, K-Pop, Hip-Hop/Rap, Acoustic

Mood-Based: Vibe/Aesthetic, TikTok/Instagram Trends

Activity: Motivational/Gym, Study/Focus, Party, Road Trip

Situational: Rainy Day, Sleep/Calm, Morning Vibes, Late Night

Cultural: Sufi/Qawwali, Spiritual/Devotional, Instrumental

🤝 Contributing
Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

⚠️ Important Notes
Webcam Required: Ensure good lighting for accurate emotion detection

API Limits: YouTube Data API has daily quota limits (10,000 units/day)

Privacy: All processing happens locally; no video data is stored or transmitted

Browser Permissions: Allow camera access when prompted by your browser

🆘 Troubleshooting
Webcam not working?

Check browser permissions

Ensure no other app is using the camera

Try refreshing the page

Verify camera drivers are updated

No songs found?

Check your YouTube API key in .env file

Verify internet connection

Try different genres or emotions

Check YouTube API quota usage

Installation issues?

Use Python 3.8+

Ensure all dependencies are installed from requirements.txt

Create and activate virtual environment

On Windows, you might need Microsoft Visual C++ Build Tools

Emotion detection inaccurate?

Ensure good lighting on your face

Position face clearly in camera view

Remove glasses or accessories that might obstruct facial features

Try in a well-lit environment

🔧 Technical Details
Emotion Model: Uses FER with MTCNN face detection

Frame Processing: Processes every 2nd frame for performance

Smoothing: Uses 15-frame history for stable emotion detection

Confidence Threshold: 40% minimum confidence for emotion classification

Real-time Performance: Optimized for 30 FPS webcam feeds

Made with ❤️ using Streamlit, TensorFlow, and Computer Vision
