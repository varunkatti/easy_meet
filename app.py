import tempfile
import os
import streamlit as st
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import pipeline
from googletrans import Translator
import pyperclip
import time

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
                text = r.recognize_google(audio_listened, language=language)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                whole_text += text
    return whole_text

# Function to get the translated summary from the audio using Google Translate
def get_translated_summary(whole_text, src_lang):
    translator = Translator()

    if src_lang != 'en':
        translated = translator.translate(whole_text, src=src_lang, dest='en')
        cleaned_text = translated.text
        return cleaned_text

    return whole_text

# Function to check if a video format is supported
def is_supported_format(filename):
    supported_formats = ['.mp4', '.avi', '.mkv']
    _, ext = os.path.splitext(filename)
    return ext in supported_formats

# Streamlit UI
st.title("Multilingual Video Summarizer")
st.write("Welcome! This is the Multilingual Video Summarizer. You can upload videos in any language (English, Hindi, or Kannada). The audio will be in the selected language, but the summary will be in English.")

video = st.file_uploader("Choose a file", type=['mp4', 'avi', 'mkv'])
button = st.button("Summarize")

# Sidebar with language selection dropdown and max/min slider
with st.sidebar:
    st.subheader("Language and Summary Length")
    lang_options = {'English': 'en', 'Hindi': 'hi', 'Kannada': 'kn'}  # ISO 639-1 codes
    selected_lang = st.selectbox('Select Language', options=list(lang_options.keys()))

    max_length = st.slider('Select max summary length', 50, 500, step=10, value=150)
    min_length = st.slider('Select min summary length', 10, 450, step=10, value=50)

# Summary generation and display
with st.spinner("Generating Summary.."):
    if button:
        if not video:
            st.error("Please upload a video before clicking the 'Summarize' button.")
        elif not is_supported_format(video.name):
            st.error("Unsupported video format. Supported formats: .mp4, .avi, .mkv")
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video.read())
            v = VideoFileClip(tfile.name)
            v.audio.write_audiofile("movie.wav")
            st.audio("movie.wav")
            whole_text = get_large_audio_transcription("movie.wav", language=lang_options[selected_lang])

            summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt")
            summarized = summarizer(whole_text, min_length=min_length, max_length=max_length, do_sample=False)
            summ = summarized[0]['summary_text']

            # Truncate the summary to fit within the selected range
            truncated_words = summ.split()[:max_length]
            if len(truncated_words) < min_length:
                truncated_words = summ.split()[:min_length]
            truncated_summary = " ".join(truncated_words)

            st.markdown("<div class='summary-container'>", unsafe_allow_html=True)
            st.write(f"📜 Video Summary ({selected_lang}):")
            st.write(whole_text)
            st.write("🌟 Translated Summary (English):")
            translated_summary = get_translated_summary(whole_text, lang_options[selected_lang])
            st.write(translated_summary)
            st.markdown("</div>", unsafe_allow_html=True)

            # Share Option
            st.markdown("<div class='summary-container'>", unsafe_allow_html=True)
            st.write("🚀 Share the Summary:")
            share_link = st.text_input("🔗 Copy and Share this Link", value="", key="share_link")
            if st.button("📋 Copy to Clipboard"):
                pyperclip.copy(translated_summary)
                st.write("Copied to clipboard!")
                time.sleep(2)  # Add a delay of 2 seconds to display the message
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.write("Developed by Vinuta. Varun")
st.markdown("</div>", unsafe_allow_html=True)
