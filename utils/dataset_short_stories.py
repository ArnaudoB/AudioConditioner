import os
from dataset_generator import extract_json, teacher, label_scene
from tqdm import tqdm
import orjson
from transformers import AutoTokenizer, AutoModelForCausalLM
from teaching_utils import TEACHER_PROMPT, MODEL_ID
import torch

SHORT_STORIES_PATH = "./data/short_stories"
OUTPUT_PATH = "./data/chunks"
MAX_CHUNK_SIZE = 100 # maximum number of words per chunk

def generate_chunks():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for short_story in os.listdir(SHORT_STORIES_PATH):
        with open(os.path.join(SHORT_STORIES_PATH, short_story), 'r', encoding='utf-8') as f:
            text = f.read()
            words = text.split()
            chunks = [' '.join(words[i:i+MAX_CHUNK_SIZE]) for i in range(0, len(words), MAX_CHUNK_SIZE)]
            for idx, chunk in enumerate(chunks):
                with open(os.path.join(OUTPUT_PATH, f"{short_story}_{idx}.txt"), 'w', encoding='utf-8') as out_f:
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
