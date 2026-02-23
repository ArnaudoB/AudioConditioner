import streamlit as st
import torch
import numpy as np
from models.AudioConditioner import AudioConditioner
from models.BLIPModel import BLIPModel
from models.CLAPModel import CLAPModel
from models.Descriptor import TwoDeepDescriptor
from models.StableAudioModel import StableAudioModel
from utils.scene_generation import generate_scene
import soundfile as sf
import io


st.set_page_config(page_title="Audio Conditioner", layout="wide")

st.title("🎵 Audio Conditioner")
st.write("Generate music from text descriptions")

# Load models (cached)
@st.cache_resource
def load_models():
    clap_model = CLAPModel()
    music_prompter = TwoDeepDescriptor(clap_dim=512, backbone_dim=256,top_p=0.1)
    music_prompter.load_state_dict(torch.load("saves/model_checkpoint.pt", map_location=torch.device('cpu')))
    conditioner = StableAudioModel()
    blip_model = BLIPModel()
    return clap_model, music_prompter, conditioner, blip_model

# Initialize scene suggestion in session state
if 'scene_suggestion' not in st.session_state:
    st.session_state.scene_suggestion = generate_scene(1, seed=None)[0]

# Input
scene_text = st.text_area(
    "Enter a scene description:",
    value=st.session_state.scene_suggestion,
    height=100
)
# Audio generation parameters
col1, col2, col3 = st.columns(3)
with col1:
    audio_end_in_s = st.slider("Audio Duration (seconds)", min_value=5, max_value=40, value=30)
with col2:
    num_waveforms_per_prompt = st.slider("Number of Waveforms", min_value=1, max_value=5, value=1)
with col3:
    num_inference_steps = st.slider("Inference Steps", min_value=10, max_value=100, value=50)

if st.button("🎬 Generate Audio"):
    with st.spinner("Loading models..."):
        clap_model, music_prompter, conditioner, blip_model = load_models()
        audio_conditioner = AudioConditioner(conditioner, music_prompter, blip_model, clap_model)
    
    with st.spinner("Generating audio..."):

        generated_audio, music_descriptor, dissimilarity_score = audio_conditioner(scene_text, 
                                                                                input_type="text", 
                                                                                audio_end_in_s=audio_end_in_s, 
                                                                                num_waveforms_per_prompt=num_waveforms_per_prompt, 
                                                                                num_inference_steps=num_inference_steps)
    # Display results
    st.success("Audio generated successfully!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Generated Prompt")
        st.write(music_descriptor.prompt())
    with col2:
        st.subheader("Negative Prompt")
        st.write(music_descriptor.negative_prompt())
    
    buffers = [io.BytesIO() for _ in range(num_waveforms_per_prompt)]
    for i in range(num_waveforms_per_prompt):
        audio = generated_audio[i].T.float().cpu().numpy()
        buffer = buffers[i]
        sf.write(buffer, audio, samplerate=44100, format='WAV')
        buffer.seek(0) # On remet le curseur au début du fichier en mémoire

    # Play audio
    st.subheader("Generated Audio")
    for i, buffer in enumerate(buffers):
        with st.container(border=True):
            st.write(f"Waveform {i+1} - Dissimilarity Score: {dissimilarity_score[i]:.4f}")
            col_audio, col_download = st.columns(2)
            with col_audio:
                st.audio(buffer, format="audio/wav")
            with col_download:
                st.download_button(
            label=f"💾 Download Audio {i+1}",
            data=buffers[i].getvalue(),
            file_name=f"generated_audio_{i+1}.wav",
            mime="audio/wav"
            )