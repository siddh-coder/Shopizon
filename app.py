from flask import Flask, render_template, request
from PIL import Image
import cv2
import yt_dlp
import os
import tempfile
from moviepy.editor import VideoFileClip
import requests
from flask_sqlalchemy import SQLAlchemy
import json

app = Flask(__name__)

# Configuration
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
headers = {"Authorization": f"Bearer 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"}

# Database setup with SQLite
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "database/app.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model for storing video analysis data
class VideoAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_url = db.Column(db.String(500), nullable=False)
    transcription = db.Column(db.Text, nullable=True)
    images = db.Column(db.Text, nullable=True)  # Image paths as a JSON list

# Function to extract images from YouTube video
def extract_images_from_youtube(video_url, num_images=5):
    ydl_opts = {
        'format': 'best',
        'outtmpl': 'temp_video.mp4',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    video = cv2.VideoCapture('temp_video.mp4')
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_images)
    extracted_images = []

    with tempfile.TemporaryDirectory() as tmpdirname:
        for i in range(num_images):
            video.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            success, frame = video.read()
            if success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                image_path = f"static/images/frame_{i}.png"
                pil_image.save(image_path)
                extracted_images.append(image_path)
        video.release()
    
    return extracted_images

# Function to extract audio and transcribe using Hugging Face API
def transcribe_audio():
    video = VideoFileClip("temp_video.mp4")
    audio_path = "temp_audio.flac"
    video.audio.write_audiofile(audio_path, codec='flac')

    with open(audio_path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    
    os.remove(audio_path)
    return response.json().get("text", "")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = request.form['video_url']
        
        # Extract images from the video
        images = extract_images_from_youtube(video_url)
        images_json = json.dumps(images)

        # Transcribe audio
        transcription_text = transcribe_audio()

        # Store analysis in SQLite database
        analysis = VideoAnalysis(video_url=video_url, transcription=transcription_text, images=images_json)
        db.session.add(analysis)
        db.session.commit()

        return render_template('index.html', images=images, result=transcription_text)

    return render_template('index.html', images=None)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
