import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union
import random
from utils.music_descriptor import MusicDescriptor


class AudioConditioner(nn.Module):
    def __init__(self,
                audio_gen_model: nn.Module,
                music_prompter: nn.Module,
                blip_model: nn.Module,
                clap_model: nn.Module,
                **args):
        super().__init__()
        self.audio_gen_model = audio_gen_model
        self.music_prompter = music_prompter
        self.blip_model = blip_model
        self.clap_model = clap_model

    def evaluate_similarity(self, generated_audio_embedding, scene_text_embedding):
        return F.cosine_similarity(generated_audio_embedding, scene_text_embedding, dim=0)
        
    def forward(self, input, input_type: str, **kwargs):
        scene_text = ""
        if input_type == "image":
            scene_text = self.blip_model(input)
        elif input_type == "text":
            scene_text = input
        
        text_embedding = self.clap_model(scene_text)
        music_descriptor = self.music_prompter(text_embedding)
        music_prompt = music_descriptor.prompt()

        generated_audio = self.audio_gen_model(music_prompt)

        dissimilarity_score = self.evaluate_similarity(self.clap_model(generated_audio), text_embedding)

        return generated_audio, music_descriptor, dissimilarity_score
