import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

class BLIPModel(torch.nn.Module):
    """Image captioning model using BLIP. Generates a descriptive sentence for a given image."""
    def __init__(self, 
                 model_id="Salesforce/blip-image-captioning-base",
                 prompt="a picture of "): 
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir="/Data/audiocond-models")
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="/Data/audiocond-models",
        )
        self.prompt = prompt

    def forward(self, image: Image.Image):
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.model.device, torch.float16)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50)

        result = self.processor.decode(outputs[0], skip_special_tokens=True)
        result = result[len(self.prompt):] if result.startswith(self.prompt) else result
            
        return result