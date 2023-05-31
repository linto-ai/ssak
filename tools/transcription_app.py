
import whisper
from audio_recorder_streamlit import audio_recorder
import streamlit as st
from linastt.infer.whisper_infer import *
import os

# This script allows the user to transcribe an audio file or a recorded audio clip using the Whisper library.

# Add custom CSS styles
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0;-m pip install --upgrade pip
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    </style>
    
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .transcription {
        border : 'bold','1px';
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 16px;
        color: #333;
    }
    </style>
    
    """,
    unsafe_allow_html=True
)



# Add content to your app
st.title('Transcriberlit')
st.write('Welcome to my Transcriberlit, This app allows you to transcribe audio files or audio clips using the Whisper library.')
st.header('Instructions')
st.write('To use this app, upload an audio file or record a clip using the microphone. Then click the "Transcribe Audio" button to transcribe the audio.')


sample_rate = 16_000

models = {
    #"ALG_small" : '/home/linagora/Transcriberlit/Model_alg/finals',
    "Tiny" : 'tiny',
    "Base" : 'base',
    "Small" : 'small',
}

Language = {
   "Arabic" : "ar",
   "English" : "en",
   "French" : "fr",
}

model_name = st.sidebar.selectbox("Select a model", list(models.keys()), index=0)
model_name = models[model_name]

language = st.sidebar.selectbox("Select a language", list(Language.keys()), index=0)
language = Language[language]

if language == "ar":
    border_style = """
    <style>
    .transcription-list {
        border: 1px solid #000;
        padding: 10px;
        border-radius: 5px;
        direction: rtl;
        text-align: right;
    }
    </style>
    """
else:
    border_style = """
    <style>
    .transcription-list {
        border: 1px solid #000;
        padding: 10px;
        border-radius: 5px;
        direction: ltr;
        text-align: left;
    }
    </style>
    """

# Upload an audio file (WAV or MP3) from the sidebar
audio_file = st.file_uploader("Upload an audio File:",type=['wav', 'mp3'])


# Record an audio clip using the microphone
audio_bytes = audio_recorder(
    energy_threshold=(-2.0, 2.0),
    pause_threshold=15.0,
    sample_rate=sample_rate,
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_size="2x",
)

# If an audio clip was recorded, write it to a file
if audio_bytes:
    filename = "audio_record.wav"
    try:
        with open(filename, "wb") as audio_file:
            audio_file.write(audio_bytes)
    except Exception as e:
        st.error(f"Error writing audio file: {e}")



# If the "Transcribe Audio" button is clicked, transcribe the audio
if st.sidebar.button('Transcribe Audio'):
    if audio_file is not None:
        st.audio(audio_file.read())
        for transcription in whisper_infer(
            model_name, audio_file.name,
            batch_size = 1,
            language = language,
            ):
            st.header('Transcription')
            if isinstance(transcription, str):
                st.markdown(border_style, unsafe_allow_html=True)
                st.markdown(f'<ul class="transcription-list">{transcription}</ul>', unsafe_allow_html=True)
            else:
                st.markdown(f'<ul class="transcription-list">{transcription}</ul>', unsafe_allow_html=True)
            

    elif audio_bytes is not None:
        with open(filename, "rb") as audio_file:
            st.audio(audio_file.read())
            for transcription in whisper_infer(
                model_name, filename,
                batch_size = 1,
                language = language,
                ):
                st.header('Transcription')
                if isinstance(transcription, str):
                    st.markdown(border_style, unsafe_allow_html=True)

                    st.markdown(f'<ul class="transcription-list">{transcription}</ul>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<ul class="transcription-list">{transcription}</ul>', unsafe_allow_html=True)

    else:
        st.error('You need to record/upload an audio file')
