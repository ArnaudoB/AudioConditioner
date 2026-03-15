# 🎵 AudioConditioner

> **Conditioned audio generation and video-to-sound synthesis using deep learning.**

AudioConditioner is a school project exploring relevant audio generation based on a prompt or an image. The project also features small animation of the input image (if any) as well as reading of the input prompt together with the background music (if asked).

---

## ✨ Features

- **Conditional audio generation** — train models that generate audio based on a prompt
- **Video-to-sound synthesis** — automatically generate sound effects synchronized to video input
- **Interactive Streamlit demos** — explore model outputs through three dedicated web interfaces

---

## 📁 Project Structure

```
AudioConditioner/
├── main.py                    # Entry point for inference / generation
├── train.py                   # Model training script
├── video_sound.py             # Video-to-sound synthesis pipeline
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

## 🎧 Demo

A sample of automatically read story is included in the repository:

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

**Keyvan Attarian, Baptiste Arnaudo** — school project

