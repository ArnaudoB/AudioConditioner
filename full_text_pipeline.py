import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union
import random
from utils.dataset import MusicDataset
from utils.music_descriptor import MusicDescriptor
from models.clap import CLAPModel
from models.Descriptor import Descriptor, TwoDeepDescriptor
from models.stable_audio import StableAudioModel
import soundfile as sf



def full_pipeline(scene_text, music_prompter: Descriptor, clap_model: CLAPModel, conditioner: StableAudioModel, device=None):

    # Step 1: Use the CLAP model to get the text embedding for the scene description
    text_embedding, _ = clap_model(texts=[scene_text], audio_waveforms=None)
    text_embedding = text_embedding.squeeze(0).to(device)
    # Step 2: Use the music prompter to generate a music descriptor from the text embedding
    #output = music_prompter(text_embedding)
    music_descriptor = music_prompter.generate_music_descriptor(text_embedding, top_p=1e-1)

    prompt = music_descriptor.prompt()
    negative_prompt = music_descriptor.negative_prompt()
    audio = conditioner(prompt, 
                       negative_prompt=negative_prompt, 
                       audio_end_in_s=30.0,
                       num_waveforms_per_prompt=1,
                       num_inference_steps=100)
    output = audio[0].T.float().cpu().numpy()

    return music_descriptor, output

def main(prompt: str):
    clap_model = CLAPModel()
    music_prompter = TwoDeepDescriptor(clap_dim=512, backbone_dim=256)
    music_prompter.load_state_dict(torch.load("saves/model_checkpoint.pt", map_location=torch.device('cpu')))
    conditioner = StableAudioModel()

    music_descriptor, audio = full_pipeline(prompt, music_prompter, clap_model, conditioner, device=torch.device('cpu'))    
    
    print("Initial prompt:", prompt)
    print("Generated music descriptor:", music_descriptor.prompt())
    print("Generated negative prompt:", music_descriptor.negative_prompt())
    sf.write("sounds/generated_audio.wav", audio, samplerate=44100)
    print("Generated audio saved to sounds/generated_audio.wav")
    
if __name__ == "__main__":
    prompt = "A man is lying on a beach at sunset, reflecting on his life and feeling a mix of nostalgia and hope for the future."
    main(prompt)