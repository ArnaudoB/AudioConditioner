import torch
import torch.nn as nn
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

try:
    import sentencepiece  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "sentencepiece is required to load the CogVideoX tokenizer. "
        "Install it with: pip install sentencepiece"
    ) from exc

class CogVideoX(nn.Module):
    """
    Encapsulation rigoureuse d'un modèle I2V Hugging Face pour une utilisation
    en tant que module autonome. Le prompt est figé dans l'état de la classe.
    """
    def __init__(
        self, 
        prompt: str, 
        model_id: str = "THUDM/CogVideoX-5b-I2V",
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        fps: int = 12,
        seed: int | None = None,
        loop_video: bool = False,
    ):
        super().__init__()
        self.prompt = prompt
        self.loop_prompt_suffix = (
            " The motion must form a seamless perfect loop: "
            "the final frame should match the first frame, "
            "continuous cyclic motion, no abrupt transition."
        )
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.fps = fps
        self.seed = seed
        self.loop_video = loop_video
        
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir="/Data/audiocond-models/"
        )
        

        self.pipe.enable_model_cpu_offload()
        
        self.pipe.vae.enable_tiling()

    def forward(self, image: Image.Image, num_inference_steps: int | None = None) -> list:
        """
        Exécute la génération de la vidéo.
        Conformément à la contrainte, cette méthode ne prend en entrée que l'image.
        
        Args:
            image (PIL.Image.Image): L'image de départ. Doit idéalement être redimensionnée 
                                     à 720x480 pour CogVideoX avant d'être passée.
            
        Returns:
            list: Une liste d'images PIL représentant les frames de la vidéo générée.
        """
        prompt = self.prompt
        if self.loop_video:
            prompt = f"{prompt}{self.loop_prompt_suffix}"

        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # Désactivation du calcul des gradients car nous sommes en phase d'inférence pure
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            )
            
        return result.frames[0]

    def generate_and_save(
        self,
        image: Image.Image,
        output_path: str = "output_video.mp4",
        num_inference_steps: int | None = None,
    ) -> list:
        """Generate a video from image and save it to disk."""
        video_frames = self.forward(image=image, num_inference_steps=num_inference_steps)
        export_to_video(video_frames, output_path, fps=self.fps)
        return video_frames

