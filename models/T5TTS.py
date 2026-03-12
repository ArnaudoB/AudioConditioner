from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Optional, Tuple

import f5_tts.api as f5_api
import numpy as np
import torch
from f5_tts.api import F5TTS
from scipy.io.wavfile import write as wav_write


class T5TTS(torch.nn.Module):
    """Simple PyTorch-style wrapper around the f5-tts API."""


    def __init__(
        self,
        ref_audio: str,
        ref_text: str,
        model: str = "F5TTS_v1_Base",
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = "/Data/audiocond-models",
    ):
        super().__init__()
        self.model_name = model
        self.engine = F5TTS(
            model=model,
            device=device,
            hf_cache_dir=hf_cache_dir,
        )
        self.ref_audio = ref_audio
        self.ref_text = ref_text



    def forward(
        self,
        prompt: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, int]:
        """Generate speech from text, optionally using a custom reference voice."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        ref_audio = reference_audio_path if reference_audio_path is not None else self.ref_audio
        ref_text = reference_text if reference_text is not None else self.ref_text

        if not ref_audio or not ref_text:
            raise ValueError("reference audio and reference text must be provided")


        wav, sample_rate, _ = self.engine.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=prompt.strip(),
            **kwargs,
        )

        if isinstance(wav, torch.Tensor):
            audio_tensor = wav.detach().float().cpu().flatten()
        else:
            audio_tensor = torch.tensor(wav, dtype=torch.float32).flatten()

        return audio_tensor, int(sample_rate)

    def synthesize_to_file(
        self,
        prompt: str,
        output_path: str,
        **kwargs: Any,
    ) -> str:
        """Generate speech and save it to a WAV file."""
        audio_tensor, sample_rate = self.forward(
            prompt=prompt,
            reference_audio_path=self.ref_audio,
            reference_text=self.ref_text,
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


    prompt = "A king and queen once upon a time reigned in a country a great way off, where there were in those days fairies. Now this king and queen had plenty of money, and plenty of fine clothes to wear, and plenty of good things to eat and drink, and a coach to ride out in every day: but though they had been married many years they had no children, and this grieved them very much indeed. But one day as the queen was walking by the side of the river, at the bottom of the garden, she saw a poor"
    ref_audio = "sounds/reference_story.mp3"  # Or provide a path to a reference audio file
    ref_text = "Walter also designed mansions, banks, churches, the hotel at Brandywine Springs, and courthouses."   # Or provide reference text describing the voice
    output_file = "demo_output.wav"
    model = T5TTS(
        ref_audio=ref_audio,
        ref_text=ref_text
    )
    saved_path = model.synthesize_to_file(
        prompt=prompt,
        output_path=output_file,
    )
    print(f"Demo generated successfully: {saved_path}")
