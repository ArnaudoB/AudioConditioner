import numpy as np
from torch.utils.data import Dataset
from utils.music_descriptor import read_music_descriptor_from_json
import tqdm

class MusicDataset(Dataset):
    """Dataset class for loading music descriptors from a JSONL file. Each line in the file should contain a JSON object with a 'scene' and a 'descriptor'."""
    def load_data_from_json(self, path_to_json: str,):
        data_list = []
        scenes = []
        with open(path_to_json, 'r') as f:
            for line in tqdm.tqdm(f, desc="Loading dataset"):
                try:
                    scene, descriptor = read_music_descriptor_from_json(line)
                    data_list.append((descriptor))
                    scenes.append(scene)
                except Exception as e:
                    print(f"Error occurred while processing line: {line}")
                    print(f"Error: {e}")
        return scenes, data_list

    def __init__(self, path_to_json: str):
        
        self.scenes, self.data_list = self.load_data_from_json(path_to_json)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        return self.scenes[idx], self.data_list[idx]
    

class EmbeddingDataset(Dataset):

    def __init__(self, music_dataset: MusicDataset, embedding_model, device=None):
        self.music_dataset = music_dataset
        self.embedding_model = embedding_model
        self.device = device
    def __len__(self):
        return len(self.music_dataset)

    def __getitem__(self, idx):
        scene, descriptor = self.music_dataset[idx]
        embedding, _ = self.embedding_model(texts=[scene], audio_waveforms=None)
        return embedding.squeeze(0), descriptor.to_differentiable_tensor(device=self.device)
    

        
if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to Python path to import utils
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.CLAPModel import CLAPModel


    dataset = MusicDataset('data/teacher_dataset.jsonl')
    print(len(dataset))
    for i in range(5):
        scene, music = dataset[i]
        print(scene, music.prompt())

    embedding_dataset = EmbeddingDataset(dataset, embedding_model=CLAPModel())
    for i in range(5):
        embedding, descriptor = embedding_dataset[i]
        print(embedding.shape, descriptor.prompt())