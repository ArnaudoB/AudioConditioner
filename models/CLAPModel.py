import torch
from transformers import ClapModel, ClapProcessor


class CLAPModel(torch.nn.Module):
    """CLAPModel is a wrapper around the ClapModel from Hugging Face's transformers library. 
    It provides methods to extract text and audio embeddings using the CLAP model. 
    The forward method can handle both text and audio inputs, returning their respective embeddings. 
    This model can be used for tasks like cross-modal retrieval or similarity computation between text and audio.
    """

    def __init__(self, model_id="laion/clap-htsat-unfused"):
        super().__init__()
        self.model = ClapModel.from_pretrained(model_id)
        self.processor = ClapProcessor.from_pretrained(model_id)
        

    def get_text_embeddings(self, texts):
        inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs).pooler_output
        return text_embeddings
    
    
    def get_audio_embeddings(self, audio_waveforms, sampling_rate=48000):
    
        # convert audio_waveforms to list of numpy arrays if it's a tensor, and ensure they are in the correct shape and dtype
        if isinstance(audio_waveforms, torch.Tensor):
            x = audio_waveforms.detach()
            if x.ndim == 3:        # [number_of_waves, channels (mono or stereo), time]
                x = x.mean(dim=1)   # -> [number_of_waves, time]
            audio_waveforms = [x[i].cpu().numpy() for i in range(x.shape[0])]
        inputs = self.processor(audio=audio_waveforms, sampling_rate=sampling_rate, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_features(**inputs).pooler_output
        return audio_embeddings
    
    def forward(self, texts, audio_waveforms=None, sampling_rate=48000):
        if texts is not None:
            text_embeddings = self.get_text_embeddings(texts)
        else:
            text_embeddings = None
        if audio_waveforms is not None:
            audio_embeddings = self.get_audio_embeddings(audio_waveforms, sampling_rate)
        else:
            audio_embeddings = None
        return text_embeddings, audio_embeddings
