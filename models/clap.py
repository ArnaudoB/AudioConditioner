import torch
from transformers import ClapModel, ClapProcessor


class CLAPModel(torch.nn.Module):
    def __init__(self, model_id="laion/clap-htsat-unfused"):
        super().__init__()
        self.model = ClapModel.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ClapProcessor.from_pretrained(model_id)

    def get_text_embeddings(self, texts):
        inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs).pooler_output
        return text_embeddings
    
    def get_audio_embeddings(self, audio_waveforms, sampling_rate=48000):
        inputs = self.processor(audio=audio_waveforms, sampling_rate=sampling_rate, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_features(**inputs).pooler_output
        return audio_embeddings
    
    def forward(self, texts, audio_waveforms, sampling_rate=48000):
        if texts is not None:
            text_embeddings = self.get_text_embeddings(texts)
        else:
            text_embeddings = None
        if audio_waveforms is not None:
            audio_embeddings = self.get_audio_embeddings(audio_waveforms, sampling_rate)
        else:
            audio_embeddings = None
        return text_embeddings, audio_embeddings
    
    

if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to Python path to import utils
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from utils import audio
    
    conditioner = CLAPModel()
    sample_texts = ["A calm and soothing piano melody", "An energetic rock guitar riff"]
    embeddings = conditioner.get_text_embeddings(sample_texts)
    print(embeddings.shape)

    audio_waveforms = [audio.audio_to_waveform("sounds/piano.mp3"), audio.audio_to_waveform("sounds/energetic_guitar.mp3")]
    audio_embeddings = conditioner.get_audio_embeddings(audio_waveforms)
    print(audio_embeddings.shape)

    for text_emb in embeddings:
        for audio_emb in audio_embeddings:
            similarity = torch.cosine_similarity(text_emb, audio_emb, dim=0)
            print(f"Similarity: {similarity.item():.4f}")

