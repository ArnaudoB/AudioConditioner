import torch.nn as nn
import torch
import torch.nn.functional as F

# Warning : we may have a problem between the sampling rate of laion vs the sampling rate of stable audio (need resampling maybe)


class AudioConditioner(nn.Module):
    """AudioConditioner is a high-level model that integrates multiple components to generate audio based on various inputs. 
    It takes an audio generation model, a descriptor model, a BLIP model for image captioning, and a CLAP model for text and audio embeddings. 
    The forward method processes the input based on its type (image or text), generates a music prompt, creates audio from the prompt, and evaluates the similarity between the generated audio and the original scene description. 
    This model can be used for tasks like generating music that matches a given scene or text description.
    """

    def __init__(self,
                audio_gen_model: nn.Module,
                descriptor: nn.Module,
                blip_model: nn.Module,
                clap_model: nn.Module,
                **args):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_gen_model = audio_gen_model.to(self.device)
        self.descriptor = descriptor.to(self.device)
        self.blip_model = blip_model.to(self.device)
        self.clap_model = clap_model.to(self.device)

        
    def forward(self, input, input_type: str, **kwargs):
        scene_text = ""
        if input_type == "image":
            scene_text = self.blip_model(input)
        elif input_type == "text":
            scene_text = input
        
        text_embedding = self.clap_model(scene_text)[0] # get text embedding from CLAP model (1, clap_dim)

        music_descriptor = self.descriptor.generate_music_descriptor(self.descriptor(text_embedding))
        music_prompt = music_descriptor.prompt()


        generated_audio = self.audio_gen_model(music_prompt) # (num_waves_per_prompt, num_channels, num_samples)

        generated_audio_embedding = self.clap_model(texts=None, audio_waveforms=generated_audio, sampling_rate=48000)[1] # get audio embedding from CLAP model (num_waveforms_per_prompt, clap_dim)

        # text_embedding is (1, clap_dim) and generated_audio_embedding is (num_waveforms_per_prompt, clap_dim), we need to compute the cosine similarity between each generated audio embedding and the text embedding, resulting in a dissimilarity score for each generated audio
        # so we change the shape
        text_embedding = text_embedding.repeat(generated_audio_embedding.shape[0], 1) # (num_waveforms_per_prompt, clap_dim)
        dissimilarity_score = F.cosine_similarity(generated_audio_embedding, text_embedding, dim=-1) # (num_waveforms_per_prompt,)

        return generated_audio, music_descriptor, dissimilarity_score

if __name__ == "__main__":

    from StableAudioModel import StableAudioModel
    from BLIPModel import BLIPModel
    from CLAPModel import CLAPModel
    from Descriptor import OneDeepDescriptor

    audio_gen_model = StableAudioModel(num_inference_steps=50, num_waveforms_per_prompt=3, seed=42)
    descriptor = OneDeepDescriptor(clap_dim=512, backbone_dim=256)
    blip_model = BLIPModel()
    clap_model = CLAPModel()

    conditioner = AudioConditioner(audio_gen_model, descriptor, blip_model, clap_model)

    sample_text = "Brutus prepares to fight against Caesar in the Roman Forum, with a tense and dramatic atmosphere."
    generated_audio, music_descriptor, dissimilarity_score = conditioner(sample_text, input_type="text")
    print("Generated audio shape:", generated_audio.shape)
    print("Dissimilarity score:", dissimilarity_score)