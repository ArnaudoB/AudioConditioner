import os
from dataset_generator import label_scene
from tqdm import tqdm
import orjson
import torch

def generate_chunks(path="./data/short_stories", output_path="./data/chunks", max_chunk_size=100):
    os.makedirs(output_path, exist_ok=True)
    for short_story in os.listdir(path):
        with open(os.path.join(path, short_story), 'r', encoding='utf-8') as f:
            text = f.read()
            words = text.split()
            chunks = [' '.join(words[i:i+max_chunk_size]) for i in range(0, len(words), max_chunk_size)]
            for idx, chunk in enumerate(chunks):
                with open(os.path.join(output_path, f"{short_story}_{idx}.txt"), 'w', encoding='utf-8') as out_f:
                    out_f.write(chunk)


def main(path_to_chunks, out_path, tokenizer=None, model=None):
    with open(out_path, "wb") as out_f:
        for chunk_file in tqdm(os.listdir(path_to_chunks), desc="Processing chunks"):
            with open(os.path.join(path_to_chunks, chunk_file), 'r', encoding='utf-8') as in_f:
                scene_text = in_f.read()
                desc = label_scene(scene_text, tokenizer=tokenizer, model=model)
                record = {
                    "scene_text": scene_text,
                    "description": desc
                }
                out_f.write(orjson.dumps(record) + b'\n')
                torch.cuda.empty_cache()
