import orjson
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

from teaching_utils import TEACHER_PROMPT, MODEL_ID

# TODO: Fix the functions (fails because has to do with the braces in the teaching prompt)


def extract_json(text: str) -> str:
    """
    TODO
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return text[start:end+1]


# Load teacher model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=torch.float16
)
model.eval()


@torch.no_grad()
def teacher(scene_text: str, max_new_tokens: int = 320, temperature: float = 0.2) -> str:
    prompt = TEACHER_PROMPT.format(scene_text=scene_text.strip())
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0], skip_special_tokens=True)
    return gen


def label_scene(scene_text: str, retries: int = 3):
    last_err = None
    for _ in range(retries):
        gen = teacher(scene_text)
        try:
            js = extract_json(gen)
            data = orjson.loads(js)
            return data
        except (ValueError, orjson.JSONDecodeError) as e:
            last_err = e
    raise RuntimeError(f"Failed to label scene after {retries} retries: {last_err}")



def main(path_to_scenes="./data/test.txt", out_path="./data/teacher_dataset.jsonl"):

    with open(path_to_scenes, "r") as f:
        scenes = [line.strip() for line in f.readlines() if line.strip()]


    with open(out_path, "wb") as f:
        for s in tqdm(scenes):
            desc = label_scene(s)
            record = {"scene": s, "descriptor": desc}
            f.write(orjson.dumps(record))
            f.write(b"\n")

if __name__ == "__main__":
    main()