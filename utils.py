# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 01:07:44 2022

@author: adeep
"""
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
def welcome():
    return "Welcome All"

def get_large_audio_transcription(path):  
    r = sr.Recognizer()
    sound = AudioSegment.from_wav(path)  
    chunks = split_on_silence(sound,
        min_silence_len = 500,
        silence_thresh = sound.dBFS-14,
        keep_silence=500,
    )
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                whole_text += text
    return whole_text
