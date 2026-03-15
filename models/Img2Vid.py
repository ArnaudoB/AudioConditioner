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
        guidance_scale: float = 6.0
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
        
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir="/Data/audiocond-models/"
        )
        

        self.pipe.enable_model_cpu_offload()
        
        self.pipe.vae.enable_tiling()

    def forward(self, image: Image.Image) -> list:
        """
        Exécute la génération de la vidéo.
        Conformément à la contrainte, cette méthode ne prend en entrée que l'image.
        
        Args:
            image (PIL.Image.Image): L'image de départ. Doit idéalement être redimensionnée 
                                     à 720x480 pour CogVideoX avant d'être passée.
            
        Returns:
            list: Une liste d'images PIL représentant les frames de la vidéo générée.
        """
        # Désactivation du calcul des gradients car nous sommes en phase d'inférence pure
        with torch.no_grad():
            result = self.pipe(
                prompt=f"{self.prompt}{self.loop_prompt_suffix}",
                image=image,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                # Fixation d'un générateur pour rendre la génération déterministe si besoin
                generator=torch.Generator(device="cpu").manual_seed(42)
            )
            
        return result.frames[0]

# ==========================================
# Exemple d'utilisation
# ==========================================
if __name__ == "__main__":
    # 1. Instanciation (le modèle est téléchargé et le prompt est fixé ici)
    prompt_texte = (
        "A photorealistic tracking shot of the subject in the image moving slowly, "
        "cinematic lighting, 4k, seamless looping animation."
    )
    model = CogVideoX(prompt=prompt_texte)
    
    # 2. Chargement de l'image (l'image doit être au format approprié, ex: 720x480)
    input_image = Image.open("pictures/skyrim.jpg").convert("RGB")
    
    # 3. Appel de la méthode forward (ne prend que l'image)
    video_frames = model(input_image)
    
    # 4. Sauvegarde
    export_to_video(video_frames, "output_video.mp4", fps=8)