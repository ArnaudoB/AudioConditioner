import torch
import numpy as np
from checkpoint_paths import SCENE_CHECKPOINT
from models.AudioConditioner import AudioConditioner
from models.BLIPModel import BLIPModel
from models.CLAPModel import CLAPModel
from models.Descriptor import TwoDeepDescriptor
from models.StableAudioModel import StableAudioModel
import soundfile as sf
import os

def load_models():
    clap_model = CLAPModel()
    music_prompter = TwoDeepDescriptor(clap_dim=512, backbone_dim=256,top_p=0.1)
    music_prompter.load_state_dict(torch.load(SCENE_CHECKPOINT, map_location=torch.device('cpu')))
    conditioner = StableAudioModel()
    blip_model = BLIPModel()
    return clap_model, music_prompter, conditioner, blip_model

def main(sample_text, audio_end_in_s, num_waveforms_per_prompt, num_inference_steps):
    clap_model, music_prompter, conditioner, blip_model = load_models()
    audio_conditioner = AudioConditioner(conditioner, music_prompter, blip_model, clap_model)

    generated_audio, music_descriptor, dissimilarity_score = audio_conditioner(sample_text, input_type="text", audio_end_in_s=audio_end_in_s, num_waveforms_per_prompt=num_waveforms_per_prompt, num_inference_steps=num_inference_steps)
    print("Generated audio shape:", generated_audio.shape)
    print("Generated prompt:", music_descriptor.prompt())
    print("Dissimilarity score:", dissimilarity_score)

    for i in range(num_waveforms_per_prompt):
        audio = generated_audio[i].T.float().cpu().numpy()
        sf.write(f"sounds/generated_audio_{i+1}.wav", audio, 48000)


if __name__ == "__main__":
    audio_end_in_s = 30.0
    num_waveforms_per_prompt = 1
    num_inference_steps = 50
    sample_text = "Brutus prepares to fight against Caesar in the Roman Forum, with a tense and dramatic atmosphere."
    main(sample_text, audio_end_in_s, num_waveforms_per_prompt, num_inference_steps)
