import io
import tempfile
from pathlib import Path

import soundfile as sf
import streamlit as st
import torch

from models.AudioConditioner import AudioConditioner
from models.BLIPModel import BLIPModel
from models.CLAPModel import CLAPModel
from models.Descriptor import TwoDeepDescriptor
from models.ReadMusic import ReadMusic
from models.StableAudioModel import StableAudioModel
from models.T5TTS import T5TTS


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT_DIR / "saves" / "model_checkpoint.pt"
DEFAULT_REFERENCE_AUDIO = ROOT_DIR / "sounds" / "reference_story.wav"
DEFAULT_REFERENCE_TEXT = (
    "Three months later, Leningrad was officially renamed Saint Petersburg."
)


st.set_page_config(page_title="ReadMusic", layout="wide")
st.title("ReadMusic")
st.write("Generate narrated speech with coherent background music from a story.")


@st.cache_resource
def load_audio_conditioner(checkpoint_path: str) -> AudioConditioner:
    clap_model = CLAPModel()
    music_prompter = TwoDeepDescriptor(clap_dim=512, backbone_dim=256, top_p=0.1)
    music_prompter.load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device("cpu"))
    )
    stable_audio = StableAudioModel()
    blip_model = BLIPModel()
    return AudioConditioner(stable_audio, music_prompter, blip_model, clap_model)


@st.cache_resource
def load_tts_engine(reference_audio_path: str, reference_text: str, device: str) -> T5TTS:
    return T5TTS(
        ref_audio=reference_audio_path,
        ref_text=reference_text,
        device=device,
    )


def save_uploaded_audio(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def tensor_to_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    audio = waveform.detach().float().cpu().numpy()

    if audio.ndim == 0:
        audio = audio.reshape(1)
    elif audio.ndim == 3:
        # Keep first candidate if a batch dimension is present.
        audio = audio[0]

    if audio.ndim == 2:
        rows = int(audio.shape[0])
        cols = int(audio.shape[-1])
        if rows <= 8 and cols > rows:
            # ReadMusic returns (channels, samples); soundfile expects (samples, channels).
            audio = audio.T

    audio = audio.astype("float32", copy=False)
    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=sample_rate, format="WAV")
    return buffer.getvalue()


def descriptor_to_text(descriptor) -> str:
    if descriptor is None:
        return ""

    prompt_value = ""
    negative_value = ""
    if hasattr(descriptor, "prompt") and callable(descriptor.prompt):
        prompt_value = str(descriptor.prompt())
    if hasattr(descriptor, "negative_prompt") and callable(descriptor.negative_prompt):
        negative_value = str(descriptor.negative_prompt())

    if negative_value:
        return f"Prompt: {prompt_value}\nNegative prompt: {negative_value}"
    return f"Prompt: {prompt_value}"


with st.sidebar:
    st.header("Model Settings")
    checkpoint_path = st.text_input("Descriptor checkpoint", value=str(DEFAULT_CHECKPOINT))
    reference_audio_path = st.text_input(
        "Reference audio path",
        value=str(DEFAULT_REFERENCE_AUDIO),
        help="Path to a local file used for voice style transfer.",
    )
    uploaded_reference = st.file_uploader(
        "Or upload reference audio",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
    )
    reference_text = st.text_area(
        "Reference transcript",
        value=DEFAULT_REFERENCE_TEXT,
        height=90,
        help="Transcript of the reference audio.",
    )

    st.divider()
    st.header("Generation Settings")
    target_sample_rate = st.select_slider(
        "Target sample rate",
        options=[16000, 22050, 24000, 32000, 44100, 48000],
        value=48000,
    )
    speech_gain = st.slider("Speech gain", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
    music_gain = st.slider("Music gain", min_value=0.0, max_value=1.5, value=0.10, step=0.05)
    fade_duration_s = st.slider("Crossfade duration (s)", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    num_waveforms_per_prompt = st.slider("Candidate waveforms", min_value=1, max_value=4, value=1)
    if num_waveforms_per_prompt > 1:
        music_index = st.slider(
            "Music candidate index",
            min_value=0,
            max_value=max(1, num_waveforms_per_prompt - 1),
            value=0,
            step=1,
        )
    else:        
        music_index = 0
    num_inference_steps = st.slider("Music inference steps", min_value=10, max_value=100, value=50)

    use_fixed_music_duration = st.checkbox("Use fixed music duration", value=False)
    fixed_music_duration = st.slider("Music duration (s)", min_value=5, max_value=45, value=30)

story_text = st.text_area(
    "Story text",
    height=280,
    placeholder="Paste your story here...",
)

if st.button("Generate ReadMusic", type="primary", use_container_width=True):
    if not story_text.strip():
        st.error("Please provide story text.")
        st.stop()

    if not Path(checkpoint_path).exists():
        st.error("Descriptor checkpoint not found.")
        st.stop()

    active_reference_audio = reference_audio_path
    if uploaded_reference is not None:
        active_reference_audio = save_uploaded_audio(uploaded_reference)

    if not Path(active_reference_audio).exists():
        st.error("Reference audio file not found. Upload one or fix the path.")
        st.stop()

    if not reference_text.strip():
        st.error("Reference transcript cannot be empty.")
        st.stop()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with st.spinner("Loading models..."):
        audio_conditioner = load_audio_conditioner(checkpoint_path)
        tts_model = load_tts_engine(active_reference_audio, reference_text, device)

    read_music_model = ReadMusic(
        audio_conditioner=audio_conditioner,
        tts_model=tts_model,
        target_sample_rate=target_sample_rate,
        speech_gain=speech_gain,
        music_gain=music_gain,
        fade_duration_s=fade_duration_s,
    )

    audio_end_in_s = fixed_music_duration if use_fixed_music_duration else None

    try:
        with st.spinner("Generating narration + music..."):
            result = read_music_model(
                text=story_text,
                reference_audio_path=active_reference_audio,
                reference_text=reference_text,
                audio_end_in_s=audio_end_in_s,
                num_waveforms_per_prompt=num_waveforms_per_prompt,
                num_inference_steps=num_inference_steps,
                music_index=music_index,
            )
    except Exception as exc:
        st.error(f"Generation failed: {exc}")
        st.stop()

    sample_rate = int(result["sample_rate"])
    merged_audio = result["merged_audio"]
    speech_audio = result["speech_audio"]
    music_audio = result["music_audio"]
    chunk_texts = result["chunks"]
    dissimilarity = result["dissimilarity_score"]

    merged_bytes = tensor_to_wav_bytes(merged_audio, sample_rate)
    speech_bytes = tensor_to_wav_bytes(speech_audio, sample_rate)
    music_bytes = tensor_to_wav_bytes(music_audio, sample_rate)

    st.success("Generation completed.")

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("Chunks", len(chunk_texts))
    with stats_col2:
        if dissimilarity.ndim == 1:
            mean_score = dissimilarity.mean().item()
        else:
            mean_score = dissimilarity[:, music_index].mean().item()
        st.metric("Mean similarity", f"{mean_score:.4f}")
    with stats_col3:
        duration_s = merged_audio.shape[-1] / float(sample_rate)
        st.metric("Duration", f"{duration_s:.1f}s")

    st.subheader("Audio")
    col_merged, col_speech, col_music = st.columns(3)
    with col_merged:
        st.write("Merged")
        st.audio(merged_bytes, format="audio/wav")
        st.download_button(
            label="Download merged",
            data=merged_bytes,
            file_name=f"readmusic_candidate_{music_index + 1}_merged.wav",
            mime="audio/wav",
        )
    with col_speech:
        st.write("Speech only")
        st.audio(speech_bytes, format="audio/wav")
        st.download_button(
            label="Download speech",
            data=speech_bytes,
            file_name=f"readmusic_candidate_{music_index + 1}_speech.wav",
            mime="audio/wav",
        )
    with col_music:
        st.write("Music only")
        st.audio(music_bytes, format="audio/wav")
        st.download_button(
            label="Download music",
            data=music_bytes,
            file_name=f"readmusic_candidate_{music_index + 1}_music.wav",
            mime="audio/wav",
        )

    st.subheader("Chunk details")
    for index, chunk in enumerate(chunk_texts, start=1):
        with st.container(border=True):
            st.write(f"Chunk {index}")
            st.write(chunk)
            if index - 1 < len(result["music_descriptor"]):
                st.caption(descriptor_to_text(result["music_descriptor"][index - 1]))
            chunk_scores = dissimilarity[index - 1]
            if chunk_scores.ndim == 0:
                chunk_scores = chunk_scores.unsqueeze(0)
            chunk_candidate_count = int(chunk_scores.shape[0])
            score_line = ", ".join(
                [f"Candidate {cand + 1}: {chunk_scores[cand].item():.4f}" for cand in range(chunk_candidate_count)]
            )
            st.write(score_line)
