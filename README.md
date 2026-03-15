# 🎵 AudioConditioner

> **Conditioned audio generation and video-to-sound synthesis using deep learning.**

AudioConditioner is a school project exploring conditional audio generation — training neural networks to produce audio guided by structured conditions, and synthesizing sound effects that match video content.

---

## ✨ Features

- **Conditional audio generation** — train models that generate audio based on conditioning signals
- **Video-to-sound synthesis** — automatically generate sound effects synchronized to video input
- **Interactive Streamlit demos** — explore model outputs through three dedicated web interfaces
- **Visualization tools** — inspect spectrograms, waveforms, and model internals
- **Modular architecture** — clean separation between data, models, training, and inference

---

## 📁 Project Structure

```
AudioConditioner/
├── main.py                    # Entry point for inference / generation
├── train.py                   # Model training script
├── video_sound.py             # Video-to-sound synthesis pipeline
├── checkpoint_paths.py        # Paths to saved model checkpoints
│
├── streamlit-readmusic.py     # 🎼 Demo: listen to generated audio
├── streamlit-video-sound.py   # 🎬 Demo: video + generated sound
├── streamlit-vis.py           # 📊 Demo: visualizations & spectrograms
│
├── models/                    # Model architecture definitions
├── utils/                     # Helper functions (audio processing, etc.)
├── data/                      # Datasets and data loading utilities
├── sounds/                    # Audio samples and references
├── saves/                     # Training checkpoints
├── report/                    # LaTeX project report (source)
├── pictures/                  # Figures and diagrams
│
├── demo_output.wav            # Sample generated audio
└── requirements.txt           # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ArnaudoB/AudioConditioner.git
cd AudioConditioner

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Usage

### Training a model

```bash
python train.py
```

### Running inference / generating audio

```bash
python main.py
```

### Video-to-sound synthesis

```bash
python video_sound.py
```

### Launching the interactive demos

```bash
# Listen to and compare generated audio samples
streamlit run streamlit-readmusic.py

# Explore video + generated sound
streamlit run streamlit-video-sound.py

# Visualize spectrograms and model outputs
streamlit run streamlit-vis.py
```

---

## 🎧 Demo

A sample generated audio file is included in the repository:

```
demo_output.wav
```

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| **Python** | Core language |
| **PyTorch** | Neural network training & inference |
| **Streamlit** | Interactive demo interfaces |
| **librosa / torchaudio** | Audio processing |
| **NumPy / SciPy** | Signal processing utilities |

---

## 📄 Report

A full written report (LaTeX source) is available in the [`report/`](./report) directory, covering the methodology, model architecture, experiments, and results.

---

## 👤 Author

**ArnaudoB** — school project

---

## 📝 License

This project was developed as part of a school curriculum. Please reach out before reusing any part of this work.
