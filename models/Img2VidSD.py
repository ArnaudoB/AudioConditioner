import torch
from torch import nn
import numpy as np
from diffusers import DiffusionPipeline, UNetSpatioTemporalConditionModel, BitsAndBytesConfig
from diffusers.utils import load_image, export_to_video

class Img2VSDModel(nn.Module):
    def __init__(self, model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt", 
                 device: str = "cuda",
                 use_4bit: bool = False):
        super().__init__()
        self.model_id = model_id
        self.device = device
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            unet = UNetSpatioTemporalConditionModel.from_pretrained(
                model_id,
                subfolder="unet",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                cache_dir="/Data/audiocond-models",
            )
        else:
            unet = UNetSpatioTemporalConditionModel.from_pretrained(
                model_id,
                subfolder="unet",
                torch_dtype=torch.float16,
                cache_dir="/Data/audiocond-models",
            )
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            unet=unet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir="/Data/audiocond-models",
            variant="fp16",
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()
    
    def forward(self, image, num_inference_steps: int = 25, height: int = 576, width: int = 1024):
        """Generate video from image.
        
        Args:
            image: Input image to animate
            num_inference_steps: Number of inference steps (default: 25)
            height: Output video height (default: 576)
            width: Output video width (default: 1024)
        """
        result = self.pipe(
            image=image,
            num_inference_steps=num_inference_steps,
            decode_chunk_size=2,
            height=height,
            width=width,
        )
        return result.frames[0]
    
    def generate_and_save(self, image, output_path: str = "output.mp4", num_inference_steps: int = 25):
        """Generate video and save to file.
        
        Args:
            image: Input image to animate
            output_path: Path where the video will be saved
            num_inference_steps: Number of inference steps for higher quality (default: 25)
        """
        video_frames = self.forward(image, num_inference_steps)
        export_to_video(video_frames, output_path)
        print(f"Video saved to {output_path}")
        return video_frames


if __name__ == "__main__":
    # Example: Animate an image with Stable Video Diffusion
    
    # Initialize the model
    model = Img2VSDModel()
    
    # Load the image
    image_path = "pictures/skyrim.jpg"
    image = load_image(image_path)
    
    # Generate animation and save to video
    output_path = "sounds/animated_video.mp4"
    model.generate_and_save(image, output_path, num_inference_steps=25)