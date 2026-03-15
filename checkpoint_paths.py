from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SAVES_DIR = ROOT_DIR / "saves"


def _resolve_checkpoint(*names: str) -> Path:
    for name in names:
        path = SAVES_DIR / name
        if path.exists():
            return path
    return SAVES_DIR / names[0]


SCENE_CHECKPOINT = _resolve_checkpoint("model_scene.pt", "model_scenes.pt")
CHUNKS_CHECKPOINT = _resolve_checkpoint("model_chunks.pt")