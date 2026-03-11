import orjson
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

from teaching_utils import TEACHER_PROMPT, MODEL_ID


def extract_json(text: str) -> str:
    """
    Extracts the first valid JSON object from the model's answer.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]

    raise ValueError("Unclosed JSON object")


@torch.no_grad()
def teacher(scene_text: str, max_new_tokens: int = 320, temperature: float = 0.2, tokenizer=None, model=None) -> str:
    """
    Prompts the teacher model with the given scene text and returns the answer.
    """
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

    gen_tokens = out[0, inputs["input_ids"].shape[1]:]   # only new tokens   # only new tokens
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return response


def label_scene(scene_text: str, retries: int = 10, tokenizer=None, model=None) -> dict:
    """
    Function to label a scene using the teacher model. It will retry to parse the JSON up to `retries` times if the model's output is not valid JSON.
    """
    last_err = None
    for _ in range(retries):
        gen = teacher(scene_text, tokenizer=tokenizer, model=model)
        try:
            js = extract_json(gen)
            data = orjson.loads(js)
            return data
        except (ValueError, orjson.JSONDecodeError) as e:
            last_err = e
    raise RuntimeError(f"Failed to label scene after {retries} retries: {last_err}")



def main(path_to_scenes="./data/generated_scenes.txt", out_path="./data/teacher_dataset.jsonl", tokenizer=None, model=None):

    with open(path_to_scenes, "r") as f:
        scenes = [line.strip() for line in f.readlines() if line.strip()]


    with open(out_path, "wb") as f:
        
        for s in tqdm(scenes):
            desc = label_scene(s, tokenizer=tokenizer, model=model)
            record = {"scene": s, "descriptor": desc}
            f.write(orjson.dumps(record))
            f.write(b"\n")

if __name__ == "__main__":
    # Load teacher model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16
    )
    model.eval()
    main(tokenizer=tokenizer, model=model)