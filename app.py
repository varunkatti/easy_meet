import tempfile
import os
import streamlit as st
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import pipeline
from langdetect import detect

# Function to convert video to audio
def video_to_audio(input_video, output_audio):
    video = VideoFileClip(input_video)
    video.audio.write_audiofile(output_audio)

# Function to get audio transcription
def get_large_audio_transcription(path, language='en-US'):
    r = sr.Recognizer()
    sound = AudioSegment.from_wav(path)
    chunks = split_on_silence(sound,
        min_silence_len=500,
        silence_thresh=sound.dBFS-14,
        keep_silence=500,
    )
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            try:
                # Detect language of the audio
                lang = detect(audio_listened.get_raw_data())
                if lang in ['hi', 'kn']:
                    text = r.recognize_google(audio_listened, language=lang)
                else:
                    text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                whole_text += text
    return whole_text

# Main app layout and styling
st.set_page_config(page_title="Video to Audio & Summary App", layout="wide", initial_sidebar_state="collapsed")
st.title("🎬 Video to Audio & Summary App 📜")
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        color: #404040;
        margin-bottom: 30px;
    }
    .file-upload-container {
        margin-bottom: 30px;
    }
    .sidebar {
        background-color: #f5f5f5;
        padding: 15px;
    }
    .summary-container {
        background-color: #ffffff;
        padding: 20px;
        border: 1px solid #dcdcdc;
        border-radius: 10px;
        margin-top: 20px;
    }
    .footer {
        font-size: 14px;
        color: #808080;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("Welcome! This is the Summary Generator. You can upload English, Hindi, or Kannada language videos to get a summary.")
video = st.file_uploader("Choose a file", type=['mp4'])
button = st.button("Summarize")

# Language selection slider
lang_options = {'English': 'en-US', 'Hindi': 'hi', 'Kannada': 'kn'}
selected_lang = st.select_slider('Select Language', options=list(lang_options.keys()))

max = st.sidebar.slider('Select max summary length', 50, 500, step=10, value=150)
min = st.sidebar.slider('Select min summary length', 10, 450, step=10, value=50)

# Summary generation and display
with st.spinner("Generating Summary.."):
    if button and video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())
        v = VideoFileClip(tfile.name)
        v.audio.write_audiofile("movie.wav")
        st.audio("movie.wav")
        whole_text = get_large_audio_transcription("movie.wav", language=lang_options[selected_lang])

        summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt")
        summarized = summarizer(whole_text, min_length=min, max_length=max, do_sample=False)
        summ = summarized[0]['summary_text']

        st.markdown("<div class='summary-container'>", unsafe_allow_html=True)
        st.write("📜 Video Summary:")
        st.write(summ)
        st.markdown("</div>", unsafe_allow_html=True)

        # Share Option
        st.markdown("<div class='summary-container'>", unsafe_allow_html=True)
        st.write("🚀 Share the Summary:")
        share_link = st.text_input("🔗 Copy and Share this Link", value=summ, key="share_link")
        st.button("📋 Copy to Clipboard", onclick=lambda: st.write(share_link))
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.write("Developed by Your Name")
st.markdown("</div>", unsafe_allow_html=True)
