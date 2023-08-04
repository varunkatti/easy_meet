import tempfile
import os
import streamlit as st
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import pipeline
from mtranslate import translate
import pyperclip

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

# Function to translate text to English using mtranslate
def translate_to_english(text):
    return translate(text, "en", "auto")

st.title("Multilingual Video Summarizer")
st.write("Welcome! This is the Multilingual Video Summarizer. You can upload videos in any language (English, Hindi, or Kannada). The audio will be in the selected language, but the summary will be in English.")
video = st.file_uploader("Choose a file", type=['mp4'])
button = st.button("Summarize")

# Sidebar with language selection dropdown and max/min slider
with st.sidebar:
    st.subheader("Language and Summary Length")
    lang_options = {'English': 'en-US', 'Hindi': 'hi', 'Kannada': 'kn'}
    selected_lang = st.selectbox('Select Language', options=list(lang_options.keys()))

    max = st.slider('Select max summary length', 50, 500, step=10, value=150)
    min = st.slider('Select min summary length', 10, 450, step=10, value=50)

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
        st.write(f"📜 Video Summary ({selected_lang}):")
        st.write(whole_text)
        st.write("🌟 Translated Summary (English):")
        translated_summary = translate_to_english(summ)
        st.write(translated_summary)
        st.markdown("</div>", unsafe_allow_html=True)

        # Share Option
        st.markdown("<div class='summary-container'>", unsafe_allow_html=True)
        st.write("🚀 Share the Summary:")
        share_link = st.text_input("🔗 Copy and Share this Link", value=translated_summary, key="share_link")
        if st.button("📋 Copy to Clipboard"):
            pyperclip.copy(share_link)
            st.write("Copied to clipboard!")
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.write("Developed by Vinuta, Varun")
st.markdown("</div>", unsafe_allow_html=True)
