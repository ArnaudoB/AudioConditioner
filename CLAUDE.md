# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AudioConditioner is a deep learning pipeline that generates context-aware background music for text scenes and images. It uses knowledge distillation: a teacher LLM (Qwen2.5-7B-Instruct) labels music descriptors, and a student MLP (TwoDeepDescriptor) learns to predict them from CLAP embeddings.

## Commands

### Inference
```bash
python main.py                          # Text → audio generation
python video_sound.py --image <path>    # Image → animated video + music
```

### Training
```bash
python train.py                         # Train student descriptor (W&B logging)
python utils/dataset_generator.py       # Generate teacher-labelled dataset
```

### Demos
```bash
streamlit run streamlit-readmusic.py    # Narrated story + background music
streamlit run streamlit-video-sound.py  # Image animation + generated music
streamlit run streamlit-vis.py          # CLAP similarity visualizations
```

### Benchmarking & Validation
```bash
python utils/benchmark.py               # CLAP cosine-similarity evaluation
python models/Descriptor.py             # Test descriptor forward pass
python models/CLAPModel.py              # Test text/audio embeddings
python utils/dataset.py                 # Test dataset loading
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Inference Pipeline
1. **Input** (text or image)
2. **Embedding**: CLAP for text; BLIP captioning → CLAP for images → 512-d embedding
3. **Descriptor**: `TwoDeepDescriptor` MLP → 13 structured music attributes (`MusicDescriptor`)
4. **Prompt construction**: `MusicDescriptor.prompt()` / `.negative_prompt()`
5. **Generation**: Stable Audio Open diffusion model (50 steps, 30s clips, 48kHz)
6. **Ranking**: CLAP cosine similarity between scene text and generated audio → best candidate returned

### Key Components

| File | Role |
|------|------|
| `models/AudioConditioner.py` | Top-level orchestrator chaining all models |
| `models/Descriptor.py` | `TwoDeepDescriptor` — multi-task MLP (classification + regression heads) |
| `models/CLAPModel.py` | Cross-modal embeddings (text ↔ audio) |
| `models/BLIPModel.py` | Image → scene caption |
| `models/StableAudioModel.py` | Diffusion-based text-to-audio wrapper |
| `models/ReadMusic.py` | Sentence-level music scheduling for long-form stories + TTS mixing |
| `utils/music_descriptor.py` | `MusicDescriptor` dataclass with 13 attributes (7 classification, 5 regression) |
| `utils/loss.py` | `AdaptedMusicDescriptorLoss` — cross-entropy for classes, MSE for regression |
| `utils/dataset.py` | `MusicDataset` + `EmbeddingDataset` (caches CLAP embeddings at load time) |
| `checkpoint_paths.py` | Resolves `saves/model_scene.pt` and `saves/model_chunks.pt` |

### Training Flow
- Dataset: `data/teacher_dataset.jsonl` — `{scene, descriptor}` pairs labelled by Qwen2.5-7B
- `EmbeddingDataset` caches CLAP embeddings at construction; 80/20 train/val split
- Checkpoints saved to `saves/` after each epoch; W&B logs batch/epoch losses

### MusicDescriptor Attributes
- **Classification** (cross-entropy): `mood`, `key_mode`, `instrumentation`, `rhythm_style`, `structure`, `production`, `dynamics_profile`
- **Regression** (MSE): `energy`, `valence`, `tempo`, `harmonic_tension`, `texture_density`
- Vocabulary defined in `utils/teaching_utils.py`
