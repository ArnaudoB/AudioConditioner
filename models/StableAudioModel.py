import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, StableAudioDiTModel, StableAudioPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel
from typing import List, Union


class StableAudioModel(torch.nn.Module):

    """
    StableAudioModel is a wrapper around the StableAudioPipeline from Hugging Face's diffusers library.
    It allows for generating audio from text prompts, with support for both single prompts and batch processing. The model uses 8-bit quantization for the text encoder and transformer to reduce memory usage while maintaining performance. The forward method can handle both single prompts and lists of prompts, making it flexible for different use cases. The generated audio is returned as a tensor.
    """

    def __init__(self, model_id="stabilityai/stable-audio-open-1.0", num_inference_steps=50, audio_end_in_s=10.0, num_waveforms_per_prompt=3, seed=42):
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

        self.num_inference_steps = num_inference_steps
        self.audio_end_in_s = audio_end_in_s
        self.num_waveforms_per_prompt = num_waveforms_per_prompt
        self.seed = seed


    def generate_audio(self, prompt, negative_prompt=None, audio_end_in_s=None, num_waveforms_per_prompt=None, num_inference_steps=None):
        """Generate audio from a single prompt"""
        generator = torch.Generator(device=self.pipeline.device).manual_seed(self.seed)
        audio = self.pipeline(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.num_inference_steps if num_inference_steps is None else num_inference_steps,
            audio_end_in_s=self.audio_end_in_s if audio_end_in_s is None else audio_end_in_s,
            num_waveforms_per_prompt=self.num_waveforms_per_prompt if num_waveforms_per_prompt is None else num_waveforms_per_prompt,
            generator=generator,
        ).audios
        return audio
    
    
    def generate_audio_batch(self, prompts: List[str], negative_prompts: Union[List[str], None] = None):
        """Generate audio from multiple prompts in batch"""
        if negative_prompts is None:
            negative_prompts = [None] * len(prompts)
        elif len(negative_prompts) != len(prompts):
            if len(negative_prompts) == 1:
                negative_prompts = negative_prompts * len(prompts)
            else:
                raise ValueError("Le nombre de negative_prompts doit être égal au nombre de prompts ou 1")
        
        generator = torch.Generator(device=self.pipeline.device).manual_seed(self.seed)
        audios = self.pipeline(
            prompts,
            negative_prompt=negative_prompts,
            num_inference_steps=self.num_inference_steps,
            audio_end_in_s=self.audio_end_in_s,
            num_waveforms_per_prompt=self.num_waveforms_per_prompt,
            generator=generator,
        ).audios
        return audios

    
    def forward(self, prompt, negative_prompt=None, audio_end_in_s=None, num_waveforms_per_prompt=None, num_inference_steps=None):
        """Forward method that can handle single prompt or list of prompts"""
        if isinstance(prompt, list):
            return self.generate_audio_batch(prompt, negative_prompt)
        else:
            return self.generate_audio(prompt, negative_prompt, audio_end_in_s, num_waveforms_per_prompt, num_inference_steps)
    
