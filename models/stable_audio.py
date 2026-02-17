import torch
import soundfile as sf
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, StableAudioDiTModel, StableAudioPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel
from typing import List, Union
import numpy as np


class StableAudioModel(torch.nn.Module):

    """
    StableAudioModel is a wrapper around the StableAudioPipeline from Hugging Face's diffusers library.
    It allows for generating audio from text prompts, with support for both single prompts and batch processing. The model uses 8-bit quantization for the text encoder and transformer to reduce memory usage while maintaining performance. The forward method can handle both single prompts and lists of prompts, making it flexible for different use cases. The generated audio is returned as a tensor.
    """

    def __init__(self, model_id="stabilityai/stable-audio-open-1.0"):
        super().__init__()

        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        text_encoder_8bit = T5EncoderModel.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            subfolder="text_encoder",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            cache_dir="/Data/audiocond-models"
        )

        quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
        transformer_8bit = StableAudioDiTModel.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            cache_dir="/Data/audiocond-models"
        )

        self.pipeline = StableAudioPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder_8bit,
            transformer=transformer_8bit,
            torch_dtype=torch.float16,
            device_map="balanced",
            cache_dir="/Data/audiocond-models"
        )

    def generate_audio(self, prompt, negative_prompt=None, num_inference_steps=200, audio_end_in_s=10.0, num_waveforms_per_prompt=3, seed=42):
        """Generate audio from a single prompt"""
        generator = torch.Generator(device=self.pipeline.device).manual_seed(seed)
        audio = self.pipeline(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            audio_end_in_s=audio_end_in_s,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            generator=generator,
        ).audios
        return audio
    
    def generate_audio_batch(self, prompts: List[str], negative_prompts: Union[List[str], None] = None, 
                           num_inference_steps=200, audio_end_in_s=10.0, num_waveforms_per_prompt=3, seed=42):
        """Generate audio from multiple prompts in batch"""
        if negative_prompts is None:
            negative_prompts = [None] * len(prompts)
        elif len(negative_prompts) != len(prompts):
            if len(negative_prompts) == 1:
                negative_prompts = negative_prompts * len(prompts)
            else:
                raise ValueError("Le nombre de negative_prompts doit être égal au nombre de prompts ou 1")
        
        generator = torch.Generator(device=self.pipeline.device).manual_seed(seed)
        audios = self.pipeline(
            prompts,
            negative_prompt=negative_prompts,
            num_inference_steps=num_inference_steps,
            audio_end_in_s=audio_end_in_s,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            generator=generator,
        ).audios
        return audios

    
    def forward(self, prompt, negative_prompt=None, num_inference_steps=200, audio_end_in_s=10.0, num_waveforms_per_prompt=3, seed=42):
        """Forward method that can handle single prompt or list of prompts"""
        if isinstance(prompt, list):
            return self.generate_audio_batch(prompt, negative_prompt, num_inference_steps, 
                                                audio_end_in_s, num_waveforms_per_prompt, seed)
        else:
            return self.generate_audio(prompt, negative_prompt, num_inference_steps, 
                                     audio_end_in_s, num_waveforms_per_prompt, seed)
    
if __name__ == "__main__":
    conditioner = StableAudioModel()
    
    # Exemple 1: Prompt unique (comportement original)
    #scene = A driven monk inside a candlelit cathedral watches a kingdom fall
    prompt = "Cinematic music. Tragic, ominous mood, negative/sad music, energetic, with a layered texture. Featuring strings, timpani, cinematic percussion. Tempo around 90 BPM, with a steady rhythm. In a minor key, with a very dissonant tension. The structure is slow build then climax, with a gradual crescendo. Duration around 30 seconds."
    negative_prompt = "Low quality."
    audio = conditioner(prompt, 
                       negative_prompt=negative_prompt, 
                       audio_end_in_s=30.0,
                       num_waveforms_per_prompt=1,
                       seed=42,
                       num_inference_steps=50)
    output = audio[0].T.float().cpu().numpy()
    sf.write("sounds/test_50.wav", output, conditioner.pipeline.vae.sampling_rate)
    
    # Exemple 2: Plusieurs prompts en parallèle (batch processing)
    prompts = [
        "The sound of rain on a window",
        "A cat meowing softly",
        "Ocean waves crashing on the shore"
    ]
    negative_prompts = ["Low quality."] * len(prompts)  # Même negative prompt pour tous
    
    print("Génération de plusieurs audios en batch...")
    audios_batch = conditioner(prompts, 
                              negative_prompt=negative_prompts,
                              audio_end_in_s=30.0,
                              num_waveforms_per_prompt=1,
                              seed=42,
                              num_inference_steps=100,
                              )  # Utilise le batch processing
    
    # Sauvegarder chaque audio
    for i, audio in enumerate(audios_batch):
        output = audio.T.float().cpu().numpy()
        sf.write(f"sounds/audio_batch_{i}.wav", output, conditioner.pipeline.vae.sampling_rate)
    
