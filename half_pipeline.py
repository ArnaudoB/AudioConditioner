import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union
import random
from utils.music_descriptor import MusicDescriptor
from models.clap import CLAPModel
from models.m_model import M_model, OneDeepM_model


def half_pipeline(scene_text, music_prompter: M_model, clap_model: CLAPModel, device=None):

    # Step 1: Use the CLAP model to get the text embedding for the scene description
    text_embedding, _ = clap_model(texts=[scene_text], audio_waveforms=None)
    text_embedding = text_embedding.squeeze(0).to(device)
    # Step 2: Use the music prompter to generate a music descriptor from the text embedding
    print(f"Text embedding shape: {text_embedding.shape}")
    print(music_prompter(text_embedding))
    music_descriptor = music_prompter.generate_music_descriptor(text_embedding)

    return music_descriptor
    
if __name__ == "__main__":
    # Example usage
    clap_model = CLAPModel()
    music_prompter = OneDeepM_model(clap_dim=512, backbone_dim=256)
    music_prompter.load_state_dict(torch.load("saves/model_checkpoint.pt", map_location=torch.device('cpu')))
    scene_text = "A driven monk inside a candlelit cathedral watches a kingdom fall."
    
    music_descriptor = half_pipeline(scene_text, music_prompter, clap_model, device=torch.device('cpu'))
    print(music_descriptor.prompt())