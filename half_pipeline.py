import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union
import random
from utils.dataset import MusicDataset
from utils.music_descriptor import MusicDescriptor
from models.clap import CLAPModel
from models.m_model import M_model, OneDeepM_model, TwoDeepM_model


def half_pipeline(scene_text, music_prompter: M_model, clap_model: CLAPModel, device=None):

    # Step 1: Use the CLAP model to get the text embedding for the scene description
    text_embedding, _ = clap_model(texts=[scene_text], audio_waveforms=None)
    text_embedding = text_embedding.squeeze(0).to(device)
    # Step 2: Use the music prompter to generate a music descriptor from the text embedding
    #output = music_prompter(text_embedding)
    music_descriptor = music_prompter.generate_music_descriptor(text_embedding, top_p=1e-1)

    return music_descriptor

def half_pipeline_with_dataset_scene(music_dataset: MusicDataset, index: int, music_prompter: M_model, clap_model: CLAPModel, device=None):
    scene_text, descriptor = music_dataset[index]
    text_embedding, _ = clap_model(texts=[scene_text], audio_waveforms=None)
    text_embedding = text_embedding.squeeze(0).to(device)
    output = music_prompter(text_embedding)
    tensor_ground_truth = descriptor.to_differentiable_tensor(device=device)
    for key in output:
        print("\n")
        print(f"{key}: {output[key]}")
        print(f"Ground truth {key}: {tensor_ground_truth[key]}")

    
if __name__ == "__main__":
    # Example usage
    clap_model = CLAPModel()
    music_prompter = TwoDeepM_model(clap_dim=512, backbone_dim=256)
    music_prompter.load_state_dict(torch.load("saves/model_checkpoint.pt", map_location=torch.device('cpu')))
    music_dataset = MusicDataset('data/teacher_dataset.jsonl')
    prompt = "A man is running through a dark alley while being chased by a mysterious figure."
    music_descriptor = half_pipeline(prompt, music_prompter, clap_model, device=torch.device('cpu'))
    print("Initial prompt:", prompt)
    print("\nGenerated music prompt:")
    print(music_descriptor.prompt())