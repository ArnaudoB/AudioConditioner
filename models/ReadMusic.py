import argparse
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.io.wavfile import write as wav_write


ROOT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_CHECKPOINT = ROOT_DIR / "saves" / "model_chunks.pt"


class ReadMusic(nn.Module):
    """Compose speech synthesis (T5-TTS) with background music (AudioConditioner)."""

    def __init__(
        self,
        audio_conditioner: nn.Module,
        tts_model: nn.Module,
        target_sample_rate: int = 48000,
        speech_gain: float = 1.0,
        music_gain: float = 0.1,
        fade_duration_s: float = 0.15,
    ):
        super().__init__()
        self.audio_conditioner = audio_conditioner
        self.tts_model = tts_model
        self.target_sample_rate = int(target_sample_rate)
        self.speech_gain = float(speech_gain)
        self.music_gain = float(music_gain)
        self.fade_duration_s = float(fade_duration_s)

    @staticmethod
    def _to_tensor_1d(waveform: torch.Tensor) -> torch.Tensor:
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        waveform = waveform.detach().float().cpu()
        if waveform.ndim == 2:
            # Convert multi-channel speech to mono.
            waveform = waveform.mean(dim=0)
        return waveform.flatten()

    @staticmethod
    def _resample_1d(waveform: torch.Tensor, src_sr: int, tgt_sr: int) -> torch.Tensor:
        if src_sr == tgt_sr:
            return waveform
        if waveform.numel() == 0:
            return waveform

        duration = waveform.numel() / float(src_sr)
        new_len = max(1, int(round(duration * float(tgt_sr))))

        batch = waveform.view(1, 1, -1)
        resampled = torch.nn.functional.interpolate(
            batch,
            size=new_len,
            mode="linear",
            align_corners=False,
        )
        return resampled.view(-1)

    @staticmethod
    def _prepare_music(music_waveform: torch.Tensor, index: int = 0) -> torch.Tensor:
        # AudioConditioner returns (num_waves, channels, samples).
        if not isinstance(music_waveform, torch.Tensor):
            music_waveform = torch.tensor(music_waveform, dtype=torch.float32)

        music_waveform = music_waveform.detach().float().cpu()
        if music_waveform.ndim == 3:
            if index >= music_waveform.shape[0]:
                raise IndexError("music_index is out of range")
            return music_waveform[index]
        if music_waveform.ndim == 2:
            return music_waveform
        if music_waveform.ndim == 1:
            return music_waveform.unsqueeze(0)
        raise ValueError("Unsupported music waveform shape")

    @staticmethod
    def _match_length(speech: torch.Tensor, music: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_len = max(speech.shape[-1], music.shape[-1])

        if speech.shape[-1] < target_len:
            speech = torch.nn.functional.pad(speech, (0, target_len - speech.shape[-1]))
        if music.shape[-1] < target_len:
            music = torch.nn.functional.pad(music, (0, target_len - music.shape[-1]))

        return speech, music

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _split_text_into_chunks(self, text: str, max_words: int = 80) -> list[str]:
        normalized_text = self._normalize_text(text)
        tokens = list(re.finditer(r"\n|[.,]|[^\s.,\n]+", normalized_text))
        if not tokens:
            return []

        chunks = []
        start_token = 0

        while start_token < len(tokens):
            words_in_chunk = 0
            last_strong_break = None
            last_comma_break = None
            current_token = start_token

            while current_token < len(tokens):
                token = tokens[current_token].group()
                if token not in {".", ",", "\n"}:
                    words_in_chunk += 1
                if token in {".", "\n"}:
                    last_strong_break = current_token
                elif token == ",":
                    last_comma_break = current_token

                if words_in_chunk >= max_words:
                    break
                current_token += 1

            if current_token >= len(tokens):
                chunk_end_char = tokens[-1].end()
                chunk = normalized_text[tokens[start_token].start():chunk_end_char].strip()
                if chunk:
                    chunks.append(chunk)
                break

            if last_strong_break is not None and last_strong_break >= start_token:
                split_token = last_strong_break
            elif last_comma_break is not None and last_comma_break >= start_token:
                split_token = last_comma_break
            else:
                split_token = current_token

            chunk_end_char = tokens[split_token].end()
            chunk = normalized_text[tokens[start_token].start():chunk_end_char].strip()
            if chunk:
                chunks.append(chunk)

            start_token = split_token + 1
            while start_token < len(tokens) and tokens[start_token].group() == "\n":
                start_token += 1

        return chunks

    @staticmethod
    def _crossfade_append(base_audio: Optional[torch.Tensor], next_audio: torch.Tensor, fade_samples: int) -> torch.Tensor:
        if base_audio is None:
            return next_audio

        overlap = min(fade_samples, base_audio.shape[-1], next_audio.shape[-1])
        if overlap <= 0:
            return torch.cat([base_audio, next_audio], dim=-1)

        fade_out = torch.linspace(1.0, 0.0, overlap, dtype=base_audio.dtype)
        fade_in = torch.linspace(0.0, 1.0, overlap, dtype=next_audio.dtype)
        blended = base_audio[:, -overlap:] * fade_out.unsqueeze(0) + next_audio[:, :overlap] * fade_in.unsqueeze(0)

        return torch.cat([base_audio[:, :-overlap], blended, next_audio[:, overlap:]], dim=-1)

    @staticmethod
    def _normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
        peak = waveform.abs().max()
        if peak > 1.0:
            waveform = waveform / peak
        return torch.clamp(waveform, -1.0, 1.0)

    def forward(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        audio_end_in_s: Optional[float] = None,
        num_waveforms_per_prompt: int = 1,
        num_inference_steps: int = 50,
        music_index: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate speech + music and return a merged waveform tensor."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        chunks = self._split_text_into_chunks(text, max_words=80)
        if not chunks:
            raise ValueError("No valid text chunk could be created from the input")

        fade_samples = max(1, int(round(self.fade_duration_s * self.target_sample_rate)))
        merged_audio = None
        full_speech_audio = None
        full_music_audio = None
        music_descriptors = []
        dissimilarity_scores = []

        print(f"Split input text into {len(chunks)} chunk(s) for sequential processing.")

        for chunk in chunks:
            print(f"Processing chunk: {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
            speech_waveform, speech_sr = self.tts_model(
                prompt=chunk,
                reference_audio_path=reference_audio_path,
                reference_text=reference_text,
                **kwargs,
            )
            speech_waveform = self._to_tensor_1d(speech_waveform)
            speech_waveform = self._resample_1d(speech_waveform, int(speech_sr), self.target_sample_rate)
            speech_duration_s = speech_waveform.shape[-1] / float(self.target_sample_rate)

            music_duration_s = speech_duration_s if audio_end_in_s is None else float(audio_end_in_s)
            music_duration_s = min(music_duration_s, 45.0)
            generated_music, music_descriptor, dissimilarity_score = self.audio_conditioner(
                chunk,
                input_type="text",
                audio_end_in_s=music_duration_s,
                num_waveforms_per_prompt=num_waveforms_per_prompt,
                num_inference_steps=num_inference_steps,
            )
            music_waveform = self._prepare_music(generated_music, index=music_index)

            speech_waveform = speech_waveform.unsqueeze(0).repeat(music_waveform.shape[0], 1)
            speech_waveform, music_waveform = self._match_length(speech_waveform, music_waveform)
            mixed = self.speech_gain * speech_waveform + self.music_gain * music_waveform

            full_speech_audio = self._crossfade_append(full_speech_audio, speech_waveform, fade_samples)
            full_music_audio = self._crossfade_append(full_music_audio, music_waveform, fade_samples)
            merged_audio = self._crossfade_append(merged_audio, mixed, fade_samples)
            music_descriptors.append(music_descriptor)
            dissimilarity_scores.append(dissimilarity_score.detach().float().cpu())

        if merged_audio is None or full_speech_audio is None or full_music_audio is None:
            raise RuntimeError("Chunked audio generation produced no audio")

        merged_audio = self._normalize_waveform(merged_audio)
        full_speech_audio = self._normalize_waveform(full_speech_audio)
        full_music_audio = self._normalize_waveform(full_music_audio)
        dissimilarity_score = torch.stack(dissimilarity_scores, dim=0)

        return {
            "merged_audio": merged_audio,
            "sample_rate": self.target_sample_rate,
            "speech_audio": full_speech_audio,
            "music_audio": full_music_audio,
            "music_descriptor": music_descriptors,
            "dissimilarity_score": dissimilarity_score,
            "chunks": chunks,
        }

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate narration + music and save merged waveform as a WAV file."""
        result = self.forward(
            text=text,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
            **kwargs,
        )
        merged = result["merged_audio"].T.detach().cpu().numpy()
        merged = np.clip(merged, -1.0, 1.0)
        merged_int16 = (merged * 32767.0).astype(np.int16)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wav_write(str(path), result["sample_rate"], merged_int16)
        return str(path)

if __name__ == "__main__":
    from AudioConditioner import AudioConditioner
    from BLIPModel import BLIPModel
    from CLAPModel import CLAPModel
    from Descriptor import TwoDeepDescriptor
    from StableAudioModel import StableAudioModel
    from T5TTS import T5TTS

    print("Paste your story below. Press Enter on an empty line to run generation.")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    story = "\n".join(lines).strip()
    if not story:
        raise ValueError("Empty story: please type at least one non-empty line.")

    print("Loading real models (this can take time)...")
    clap_model = CLAPModel()
    music_prompter = TwoDeepDescriptor(clap_dim=512, backbone_dim=256, top_p=0.1)
    music_prompter.load_state_dict(torch.load(CHUNKS_CHECKPOINT, map_location=torch.device("cpu")))
    stable_audio = StableAudioModel()
    blip_model = BLIPModel()
    audio_conditioner = AudioConditioner(stable_audio, music_prompter, blip_model, clap_model)
    ref_audio = "sounds/reference_story.wav"  
    ref_text = "Three mounth later, Leningrad was officially renamed Saint Petersburg."  
    tts = T5TTS(ref_audio=ref_audio, ref_text=ref_text)

    model = ReadMusic(
        audio_conditioner=audio_conditioner,
        tts_model=tts,
        target_sample_rate=48000,
    )

    print("Generating merged narration + music...")
    saved_path = model.synthesize_to_file(
        text=story,
        output_path="sounds/result.wav",
    )

    print("Full pipeline test passed.")
    print("Output file:", saved_path)
