from diffusers import StableDiffusionPipeline
import torch
from torch import nn


class StableDiffusionModel(nn.Module):
    """Wrapper around Stable Diffusion for text-to-image generation."""

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir="/Data/audiocond-models"
        )
        self.pipe = self.pipe.to(device)
    
    def forward(self, prompt: str):
        """Generate image from prompt."""
        result = self.pipe(prompt)
        return result.images[0]
    
    def generate_and_save(self, prompt: str, output_path: str = "pictures/generated_image.png"):
        """Generate image and save to file."""
        image = self(prompt)
        image.save(output_path)
        print(f"Image saved to {output_path}")
        return image