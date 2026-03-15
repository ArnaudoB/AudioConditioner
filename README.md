# AudioConditioner

AudioConditioner is a deep-learning pipeline that automatically generates context-aware background music for text scenes or images. Given a story excerpt or picture, it produces a structured music descriptor (mood, tempo, instrumentation, etc.) and synthesises a matching audio clip using Stable Audio.

The project also includes a **ReadMusic** feature that combines narrated text-to-speech with coherent background music, and a **Video + Music** generator that animates a still image with CogVideoX and scores it automatically.

---

## How it works

```
Text / Image
     │
     ▼
  CLAP / BLIP          ← embed scene into a 512-d vector
     │
     ▼
 TwoDeepDescriptor     ← lightweight MLP trained via knowledge distillation
     │                    from Qwen2.5-7B-Instruct (the "teacher")
     ▼
 MusicDescriptor       ← structured output:
     │                    mood, energy, valence, tempo, key/mode,
     │                    instrumentation, rhythm style, structure,
     │                    production style, dynamics profile, …
     ▼
 Stable Audio Open     ← text-to-audio diffusion model (50 steps, 30 s)
     │
     ▼
  .wav file
```

**Training (knowledge distillation)**
The teacher model (Qwen2.5-7B-Instruct) labels story chunks with structured `MusicDescriptor` JSON objects. The student `TwoDeepDescriptor` is then trained with `AdaptedMusicDescriptorLoss` (cross-entropy for categorical attributes, MSE for continuous ones) to replicate the teacher's outputs directly from CLAP embeddings, avoiding LLM inference at runtime.

---

## Project structure

```
AudioConditioner/
├── main.py                    # CLI inference: text → .wav
├── train.py                   # Student descriptor training (W&B logging)
├── video_sound.py             # Image → animated video + music pipeline
│
├── streamlit-readmusic.py     # Demo: narrated story + background music
├── streamlit-video-sound.py   # Demo: image animation + generated music
├── streamlit-vis.py           # Demo: CLAP similarity visualisations
│
├── models/
│   ├── AudioConditioner.py    # Top-level model (BLIP/CLAP + Descriptor + Stable Audio)
│   ├── Descriptor.py          # Multi-task MLP: Descriptor, OneDeepDescriptor, TwoDeepDescriptor
│   ├── CLAPModel.py           # CLAP text & audio embeddings
│   ├── BLIPModel.py           # Image captioning (image → scene text)
│   ├── StableAudioModel.py    # Stable Audio Open wrapper
│   ├── ReadMusic.py           # Chunk-level music scheduling for long stories
│   ├── T5TTS.py               # F5-TTS text-to-speech wrapper
│   ├── ParlerTTS.py           # Alternative TTS backend
│   └── Img2VidSD.py           # CogVideoX / Stable Diffusion video generation
│
├── utils/
│   ├── dataset_generator.py   # Teacher labelling: scene text → MusicDescriptor JSON
│   ├── teaching_utils.py      # Vocabulary lists & teacher prompt
│   ├── dataset.py             # MusicDataset / EmbeddingDataset
│   ├── dataset_short_stories.py  # Story chunking utilities
│   ├── music_descriptor.py    # MusicDescriptor dataclass & prompt builder
│   ├── benchmark.py           # CLAP-similarity benchmark (LLM / random / AC)
│   ├── loss.py                # MSEMusicDescriptorLoss, AdaptedMusicDescriptorLoss
│   ├── audio_utils.py         # Waveform helpers
│   └── scene_generation.py    # Scene text utilities
│
├── data/
│   ├── teacher_dataset.jsonl      # Full labelled dataset
│   ├── teacher_dataset_chunks.jsonl
│   ├── chunks/                    # Story chunks used for training
│   ├── chunks_benchmark/          # Held-out chunks for evaluation
│   └── short_stories/             # Raw story source files
│
├── saves/                     # Model checkpoints (.pt)
├── sounds/                    # Reference and generated audio
├── pictures/                  # Sample images for video demo
├── report/                    # LaTeX source for the project report
└── requirements.txt
```

---

## Getting started

### Prerequisites

- Python 3.10+
- CUDA GPU recommended (Stable Audio + CogVideoX are GPU-intensive)

### Installation

```bash
git clone https://github.com/ArnaudoB/AudioConditioner.git
cd AudioConditioner
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Generate audio from a text scene

```bash
python main.py
```

Edit the `sample_text`, `audio_end_in_s`, and `num_inference_steps` variables at the bottom of [main.py](main.py). Output files are written to `sounds/`.

### Train the student descriptor

```bash
python train.py
```

Requires `data/teacher_dataset.jsonl`. Training is logged to Weights & Biases. The checkpoint is saved to `saves/model_checkpoint.pt`.

### Generate the teacher-labelled dataset

```python
from utils.dataset_generator import label_scene
# label a chunk of story text → returns a MusicDescriptor dict
```

The full dataset generation pipeline lives in `utils/dataset_generator.py`. It uses Qwen2.5-7B-Instruct loaded locally.

### Run the benchmark

```bash
python utils/benchmark.py
```

Computes CLAP cosine-similarity scores for three generation strategies (AudioConditioner, LLM baseline, random baseline) and saves histograms + JSON results to `report/`.

### Streamlit demos

```bash
# Narrated story + background music
streamlit run streamlit-readmusic.py

# Animate an image and score it with music
streamlit run streamlit-video-sound.py

# CLAP similarity visualisations
streamlit run streamlit-vis.py
```

---

## Model details

### TwoDeepDescriptor

A two-hidden-layer MLP sitting on top of a 512-dimensional CLAP embedding. It has 13 output heads:

| Attribute | Type | Vocabulary / Range |
|---|---|---|
| mood | multi-label classification | 20 moods |
| energy | regression | [0, 1] |
| valence | regression | [0, 1] |
| tempo | regression | [50, 180] BPM |
| key_mode | classification | major / minor / ambiguous |
| harmonic_tension | regression | [0, 1] |
| texture_density | regression | [0, 1] |
| instrumentation | multi-label classification | 23 instruments |
| rhythm_style | classification | 7 styles |
| structure | classification | 5 structures |
| production_style | multi-label classification | 10 styles |
| dynamics_profile | classification | 5 profiles |

At inference, multi-label heads use top-p sampling (default `top_p=0.1`).

### ReadMusic

Splits a long story into sentence-level chunks, generates a music descriptor per chunk, and schedules audio clips to create a coherent, continuously evolving soundtrack. Narration is produced with F5-TTS (or ParlerTTS as fallback) and mixed with the background music.

---

## Tech stack

| Component | Library |
|---|---|
| Scene embeddings | `laion/clap-htsat-unfused` (CLAP) |
| Image captioning | BLIP (Salesforce) |
| Music generation | Stable Audio Open (`stabilityai/stable-audio-open-1.0`) |
| Teacher LLM | Qwen2.5-7B-Instruct |
| Video generation | CogVideoX |
| TTS | F5-TTS / T5 |
| Training | PyTorch + W&B |
| Demos | Streamlit |

---

## Report

The full written report (LaTeX source) is in [`report/`](./report), covering model architecture, the knowledge-distillation training procedure, benchmark results, and applications.

---

## Author

**Keyvan Attarian, Baptiste Arnaudo** — school project

