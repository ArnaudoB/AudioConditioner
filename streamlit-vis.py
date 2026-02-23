import streamlit as st
import torch
import numpy as np
from full_text_pipeline import full_pipeline
from models.clap import CLAPModel
from models.Descriptor import TwoDeepDescriptor
from models.stable_audio import StableAudioModel
import soundfile as sf
import os
import io

st.set_page_config(page_title="Audio Conditioner", layout="wide")

st.title("🎵 Audio Conditioner")
st.write("Generate music from text descriptions")

# Load models (cached)
@st.cache_resource
def load_models():
    clap_model = CLAPModel()
    music_prompter = TwoDeepDescriptor(clap_dim=512, backbone_dim=256)
    music_prompter.load_state_dict(torch.load("saves/model_checkpoint.pt", map_location=torch.device('cpu')))
    conditioner = StableAudioModel()
    return clap_model, music_prompter, conditioner

# Input
scene_text = st.text_area(
    "Enter a scene description:",
    value="A teacher is giving a lecture in a classroom, with students attentively listening and taking notes.",
    height=100
)

if st.button("🎬 Generate Audio"):
    with st.spinner("Loading models..."):
        clap_model, music_prompter, conditioner = load_models()
    
    with st.spinner("Generating audio..."):
        music_descriptor, audio = full_pipeline(
            scene_text, 
            music_prompter, 
            clap_model, 
            conditioner, 
            device=torch.device('cpu')
        )
    
    # Display results
    st.success("Audio generated successfully!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Generated Prompt")
        st.write(music_descriptor.prompt())
    with col2:
        st.subheader("Negative Prompt")
        st.write(music_descriptor.negative_prompt())
    
    # Play audio
    st.subheader("Generated Audio")
    buffer = io.BytesIO()
    
    # Écriture de l'array NumPy dans le buffer au format WAV
    sf.write(buffer, audio, samplerate=44100, format='WAV')
    buffer.seek(0) # On remet le curseur au début du fichier en mémoire

    # Play audio
    st.subheader("Generated Audio")
    st.audio(buffer, format="audio/wav")
    
    # Save option
    st.download_button(
        label="💾 Download Audio",
        data=buffer.getvalue(),
        file_name="generated_audio.wav",
        mime="audio/wav"
    )