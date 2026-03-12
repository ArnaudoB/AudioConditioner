from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from video_sound import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    GenerationResult,
    generate_video_with_music_from_image,
    load_models,
)


ROOT_DIR = Path(__file__).resolve().parent
PICTURES_DIR = ROOT_DIR / "pictures"


st.set_page_config(page_title="Video + Music Generator", layout="wide")
st.title("Video + Music Generator")
st.write("Anime une image avec Stable Video Diffusion et génère une musique cohérente avec AudioConditioner.")


@st.cache_resource
def get_models(checkpoint_path: str):
    return load_models(Path(checkpoint_path))


def list_sample_images() -> list[str]:
    if not PICTURES_DIR.exists():
        return []
    return sorted(
        path.name
        for path in PICTURES_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def load_selected_image(uploaded_file, sample_name: str | None) -> tuple[Image.Image | None, str | None]:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        return image, Path(uploaded_file.name).stem
    if sample_name:
        sample_path = PICTURES_DIR / sample_name
        image = Image.open(sample_path).convert("RGB")
        return image, sample_path.stem
    return None, None


def display_result(result: GenerationResult):
    final_video_bytes = result.final_video_path.read_bytes()
    audio_bytes = result.audio_path.read_bytes()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Video final")
        st.video(final_video_bytes, format="video/mp4")
    with col2:
        st.subheader("Musique")
        st.audio(audio_bytes, format="audio/wav")

    st.subheader("Prompts")
    prompt_col, negative_col = st.columns(2)
    with prompt_col:
        st.write("Prompt généré")
        st.write(result.prompt)
    with negative_col:
        st.write("Negative prompt")
        st.write(result.negative_prompt)

    st.metric("Similarity score", f"{result.similarity_score:.4f}")

    download_col1, download_col2, download_col3 = st.columns(3)
    with download_col1:
        st.download_button(
            label="Télécharger la vidéo finale",
            data=result.final_video_path.read_bytes(),
            file_name=result.final_video_path.name,
            mime="video/mp4",
        )
    with download_col2:
        st.download_button(
            label="Télécharger la vidéo muette",
            data=result.silent_video_path.read_bytes(),
            file_name=result.silent_video_path.name,
            mime="video/mp4",
        )
    with download_col3:
        st.download_button(
            label="Télécharger l'audio",
            data=result.audio_path.read_bytes(),
            file_name=result.audio_path.name,
            mime="audio/wav",
        )

    st.caption(f"Fichiers sauvegardés dans {result.final_video_path.parent}")


sample_images = list_sample_images()
source_mode = st.radio("Source de l'image", ["Image locale", "Image du dossier pictures"], horizontal=True)

uploaded_file = None
selected_sample = None
if source_mode == "Image locale":
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png", "webp"])
else:
    if sample_images:
        selected_sample = st.selectbox("Choisir une image d'exemple", sample_images)
    else:
        st.warning("Aucune image trouvée dans le dossier pictures.")

image, image_stem = load_selected_image(uploaded_file, selected_sample)
if image is not None:
    st.image(image, caption=image_stem, use_container_width=True)

st.subheader("Paramètres")
col1, col2, col3 = st.columns(3)
with col1:
    audio_duration = st.slider("Durée audio (secondes)", min_value=5, max_value=40, value=30)
with col2:
    audio_steps = st.slider("Steps audio", min_value=10, max_value=100, value=50)
with col3:
    video_steps = st.slider("Steps vidéo", min_value=10, max_value=40, value=25)

num_waveforms = st.slider("Nombre d'audios candidats", min_value=1, max_value=4, value=1)
checkpoint_path = st.text_input("Checkpoint descriptor", value=str(DEFAULT_CHECKPOINT))
output_dir = st.text_input("Dossier de sortie", value=str(DEFAULT_OUTPUT_DIR))

if st.button("Générer vidéo + musique", type="primary", use_container_width=True):
    if image is None or image_stem is None:
        st.error("Choisis une image avant de lancer la génération.")
    elif not Path(checkpoint_path).exists():
        st.error("Le checkpoint indiqué n'existe pas.")
    else:
        run_name = f"{image_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with st.spinner("Chargement des modèles..."):
            audio_conditioner, img2vid_model, stable_audio_model = get_models(checkpoint_path)
        with st.spinner("Génération de la vidéo et de la musique..."):
            result = generate_video_with_music_from_image(
                image=image,
                image_stem=run_name,
                output_dir=Path(output_dir),
                audio_conditioner=audio_conditioner,
                img2vid_model=img2vid_model,
                stable_audio_model=stable_audio_model,
                audio_duration=audio_duration,
                audio_inference_steps=audio_steps,
                video_inference_steps=video_steps,
                num_waveforms_per_prompt=num_waveforms,
            )
        st.session_state["last_result"] = {
            "prompt": result.prompt,
            "negative_prompt": result.negative_prompt,
            "similarity_score": result.similarity_score,
            "silent_video_path": str(result.silent_video_path),
            "audio_path": str(result.audio_path),
            "final_video_path": str(result.final_video_path),
        }
        st.success("Génération terminée.")
        display_result(result)

if "last_result" in st.session_state:
    saved = st.session_state["last_result"]
    restored_result = GenerationResult(
        prompt=saved["prompt"],
        negative_prompt=saved["negative_prompt"],
        similarity_score=saved["similarity_score"],
        silent_video_path=Path(saved["silent_video_path"]),
        audio_path=Path(saved["audio_path"]),
        final_video_path=Path(saved["final_video_path"]),
    )
    st.divider()
    st.subheader("Dernier résultat")
    display_result(restored_result)