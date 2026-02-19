import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class BLIPModel(torch.nn.Module):
    """BLIPModel is a wrapper around the BLIPModel from Hugging Face's transformers library. 
    It provides methods to extract text and image embeddings using the BLIP model. 
    The forward method can handle both text and image inputs, returning their respective embeddings. 
    This model can be used for tasks like cross-modal retrieval or similarity computation between text and images.
    """
    def __init__(self, model_id="Salesforce/blip-image-captioning-base"):
        super().__init__()
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id)

    def forward(self, input):
        # Process the input (image) and return the corresponding caption
        inputs = self.processor(input, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.processor.decode(outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)