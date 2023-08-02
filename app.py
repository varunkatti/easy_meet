import streamlit as st
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import pipeline

def video_to_audio(input_video, output_audio):
    video = mp.VideoFileClip(input_video)
    video.audio.write_audiofile(output_audio)

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
        chunk_filename = f"chunk{i}.wav"
        audio_chunk.export(chunk_filename, format="wav")
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            try:
                if language in ('hi', 'kn'):  # For Hindi and Kannada
                    text = r.recognize_google(audio_listened, language=language)
                else:
                    text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                whole_text += text
    return whole_text

def generate_summary(text, max_length=150, min_length=50):
    summarizer = pipeline("summarization", model="facebook/mbart-large-cc25")
    summarized = summarizer(text, min_length=min_length, max_length=max_length, src_lang="hi_IN")
    return summarized[0]['summary_text']

def main():
    st.title("Video to Audio & Summary App")
    video = st.file_uploader("Upload a video", type=["mp4"])
    button = st.button("Summarize")

    lang_options = {'English': 'en-US', 'Hindi': 'hi', 'Kannada': 'kn'}
    selected_lang = st.sidebar.selectbox("Select Language", list(lang_options.keys()))

    max_length = st.sidebar.slider('Select max summary length', 50, 500, step=10, value=150)
    min_length = st.sidebar.slider('Select min summary length', 10, 450, step=10, value=50)

    if button and video:
        with st.spinner("Converting video to audio..."):
            tfile = video.name
            video_to_audio(tfile, "output_audio.wav")

        with st.spinner("Generating audio transcription..."):
            whole_text = get_large_audio_transcription("output_audio.wav", language=lang_options[selected_lang])

        with st.spinner("Generating summary..."):
            video_summary = generate_summary(whole_text, max_length, min_length)

        st.write("Video Summary:")
        st.write(video_summary)

if __name__ == "__main__":
    main()
