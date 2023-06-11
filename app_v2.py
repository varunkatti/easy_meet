# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 01:04:50 2022

@author: adeep
"""
import cv2 as cv
import tempfile
import numpy as np
import pandas as pd
import streamlit as st 
import joblib
import os
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import transformers
from transformers import pipeline
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import sent_tokenize
import re
from utils import welcome, get_large_audio_transcription

from PIL import Image


def main():
 
    
    st.title("Summarize Text")
    video = st.file_uploader("Choose a file", type=['mp4'])
    button = st.button("Summarize")
    
    max = st.sidebar.slider('Select max', 50, 500, step=10, value=150)
    min = st.sidebar.slider('Select min', 10, 450, step=10, value=50)
    with st.spinner("Generating Summary.."):
        if button and video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video.read())
            #st.write(tfile.name)
            v = VideoFileClip(tfile.name)
            v.audio.write_audiofile("movie.wav")
            #st.video(video, format="video/mp4", start_time=0)
            st.audio("movie.wav")
            whole_text=get_large_audio_transcription("movie.wav")
            #st.write(whole_text)
            summarizer = pipeline("summarization")
            summarized = summarizer(whole_text, min_length=75, max_length=300)
            summ=summarized[0]['summary_text']
            st.write(summ)
            

if __name__ == '__main__':
    
    main()
