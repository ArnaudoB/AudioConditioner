from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import torch
from scipy.io.wavfile import write as wav_write
from transformers import pipeline


class ParlerTTS(torch.nn.Module):
    
    """
    ParlerTTS is a wrapper around the Hugging Face pipeline for text-to-speech generation using the "parler-tts/parler-tts-mini-v1" model. It provides a simple interface for generating audio from text prompts. The forward method takes a text prompt as input and returns the generated audio as a tensor. The model is designed to be used in a PyTorch training loop, allowing for easy integration with other components of an audio generation system.
    """

    def __init__(
        self,
        model_id: str = "parler-tts/parler-tts-large-v1",
    ):
        super().__init__()
        self.model_id = model_id
        self.active_model_id = model_id
        self.default_voice_description = (
            "An expressive English audiobook narrator voice with clear articulation, "
            "warm tone, steady pace, and natural pauses between sentences."
        )

        device = 0 if torch.cuda.is_available() else -1
        tts_task = cast(Any, "text-to-speech")
        self.pipe = pipeline(
            tts_task,
            model=self.model_id,
            device=device,
            trust_remote_code=True,
            cache_dir="/Data/audiocond-models"
        )

    def _run_pipeline(self, prompt: str, description: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate audio with a single clear call path."""
        try:
            if self.active_model_id.startswith("parler-tts"):
                out = self.pipe(prompt, description=description, **kwargs)
            else:
                out = self.pipe(prompt, **kwargs)
        except Exception as err:  # noqa: BLE001
            raise RuntimeError(f"Failed to generate speech with {self.active_model_id}: {err}") from err

        if isinstance(out, list):
            out = out[0]
        if not isinstance(out, dict) or "audio" not in out:
            raise RuntimeError(f"Failed to generate speech with {self.active_model_id}: unexpected pipeline output")

        return out

    def forward(
        self,
        prompt: str,
        voice_description: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, int]:
        """Generate waveform from a text prompt and return (audio, sample_rate)."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        description = voice_description or self.default_voice_description
        output = self._run_pipeline(prompt.strip(), description=description, **kwargs)

        audio = output["audio"]
        sample_rate = int(output.get("sampling_rate", 24000))

        if isinstance(audio, torch.Tensor):
            audio_tensor = audio.detach().float().cpu().flatten()
        else:
            audio_tensor = torch.tensor(audio, dtype=torch.float32).flatten()

        return audio_tensor, sample_rate

    def synthesize_to_file(
        self,
        prompt: str,
        output_path: str,
        voice_description: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate speech and save it to a wav file."""
        audio_tensor, sample_rate = self.forward(
            prompt,
            voice_description=voice_description,
            **kwargs,
        )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        waveform = audio_tensor.detach().cpu().numpy()
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform_int16 = (waveform * 32767.0).astype(np.int16)
        wav_write(str(path), sample_rate, waveform_int16)
        return str(path)


if __name__ == "__main__":
    demo_text = (
        "The rain tapped softly against the library window while she turned the final page. "
        "A quiet breath escaped her lips, and the old clock marked midnight in the hall. "
        "In that stillness, every memory felt alive again."
    )

    model = ParlerTTS()
    output_file = "sounds/parler_tts_novel_demo.wav"

    try:
        print(f"Using TTS model: {model.active_model_id}")
        saved_path = model.synthesize_to_file(
            prompt=demo_text,
            output_path=output_file,
        )
        print(f"Demo generated successfully: {saved_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Demo generation failed: {exc}")



