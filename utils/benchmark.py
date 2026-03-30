"""Benchmarking: compares CLAP cosine similarity across generation methods (LLM, random, AudioConditioner)."""

import json

from dataset_short_stories import generate_chunks
from models.AudioConditioner import AudioConditioner
from models.StableAudioModel import StableAudioModel
from models.BLIPModel import BLIPModel
from models.CLAPModel import CLAPModel
from models.Descriptor import TwoDeepDescriptor
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
from dataset_generator import label_scene
from music_descriptor import MusicDescriptor
from torch.nn import functional as F
from teaching_utils import MOOD_LIST, INSTRUMENTATION_LIST, RHYTHM_STYLE_LIST, STRUCTURE_LIST, PRODUCTION_STYLE_LIST, DYNAMICS_PROFILE_LIST, KEY_MODE_LIST, TEMPO_RANGE
import matplotlib.pyplot as plt

def benchmark_chunk_generation(path, output_path, max_chunk_size):
    """Generate text chunks from short stories for evaluation."""
    generate_chunks(path=path, output_path=output_path, max_chunk_size=max_chunk_size)


def scores_llm_generation(path_to_chunks, save_path="./report/llm_generation_scores.json"):
    """Evaluate CLAP cosine similarity using teacher LLM-generated music descriptors."""
    clap = CLAPModel()
    audio_gen_model = StableAudioModel(num_inference_steps=50, num_waveforms_per_prompt=1, seed=42)
    clap.eval()
    audio_gen_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        device_map="auto",
        dtype=torch.float16
    )
    model.eval()
    scores = []
    for chunk_file in tqdm(os.listdir(path_to_chunks)[:100], desc="Benchmarking LLM generation"):
        with open(os.path.join(path_to_chunks, chunk_file), 'r', encoding='utf-8') as in_f:
            scene_text = in_f.read()
            llm_json_output = label_scene(scene_text, tokenizer=tokenizer, model=model)
            music_descriptor = MusicDescriptor(
                mood=llm_json_output["mood"],
                energy=llm_json_output["energy"],
                valence=llm_json_output["valence"],
                tempo=llm_json_output["tempo_bpm"],
                key_mode=llm_json_output["key_mode"],
                harmonic_tension=llm_json_output["harmonic_tension"],
                texture_density=llm_json_output["texture_density"],
                instrumentation=llm_json_output["instrumentation"],
                rhythm_style=llm_json_output["rhythm_style"],
                structure=llm_json_output["structure"],
                production_style=llm_json_output["production_style"],
                dynamics_profile=llm_json_output["dynamics_profile"]
            )
            prompt = music_descriptor.prompt()
            generated_audio = audio_gen_model(prompt, audio_end_in_s=30, num_waveforms_per_prompt=1, num_inference_steps=50)
            generated_audio_embedding = clap(texts=None, audio_waveforms=generated_audio, sampling_rate=48000)[1] # get audio embedding from CLAP model
            text_embedding = clap(scene_text)[0] # get text embedding from CLAP model
            dissimilarity_score = F.cosine_similarity(generated_audio_embedding, text_embedding.repeat(generated_audio_embedding.shape[0], 1), dim=-1)
            scores.append({
                "scene_text": scene_text,
                "dissimilarity_score": dissimilarity_score.cpu().numpy().tolist()[0]
            })
            
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=4)
        print(f"Saved LLM generation scores to {save_path}")
    return scores

def scores_random_generation(path_to_chunks, save_path="./report/random_generation_scores.json"):
    """Evaluate CLAP cosine similarity using randomly sampled music descriptors (baseline)."""
    clap = CLAPModel()
    audio_gen_model = StableAudioModel(num_inference_steps=50, num_waveforms_per_prompt=1, seed=42)
    clap.eval()
    audio_gen_model.eval()

    scores = []
    for chunk_file in tqdm(os.listdir(path_to_chunks)[:100], desc="Benchmarking random generation"):
        with open(os.path.join(path_to_chunks, chunk_file), 'r', encoding='utf-8') as in_f:
            scene_text = in_f.read()
            random_dictionary = {
                "mood": [MOOD_LIST[i] for i in torch.randint(0, len(MOOD_LIST), (torch.randint(1, 4, (1,)).item(),))],
                "energy": torch.rand(1).item(),
                "valence": torch.rand(1).item(),
                "tempo": torch.randint(TEMPO_RANGE[0], TEMPO_RANGE[1], (1,)).item(),
                "key_mode": KEY_MODE_LIST[torch.randint(0, len(KEY_MODE_LIST), (1,)).item()],
                "harmonic_tension": torch.rand(1).item(),
                "texture_density": torch.rand(1).item(),
                "instrumentation": [INSTRUMENTATION_LIST[i] for i in torch.randint(0, len(INSTRUMENTATION_LIST), (torch.randint(1, 5, (1,)).item(),))],
                "rhythm_style": RHYTHM_STYLE_LIST[torch.randint(0, len(RHYTHM_STYLE_LIST), (1,)).item()],
                "structure": STRUCTURE_LIST[torch.randint(0, len(STRUCTURE_LIST), (1,)).item()],
                "production_style": [PRODUCTION_STYLE_LIST[i] for i in torch.randint(0, len(PRODUCTION_STYLE_LIST), (torch.randint(0, 4, (1,)).item(),))],
                "dynamics_profile": DYNAMICS_PROFILE_LIST[torch.randint(0, len(DYNAMICS_PROFILE_LIST), (1,)).item()]
            }
            random_music_descriptor = MusicDescriptor(**random_dictionary)
            prompt = random_music_descriptor.prompt()
            generated_audio = audio_gen_model(prompt, audio_end_in_s=30, num_waveforms_per_prompt=1, num_inference_steps=50)
            generated_audio_embedding = clap(texts=None, audio_waveforms=generated_audio, sampling_rate=48000)[1] # get audio embedding from CLAP model
            text_embedding = clap(scene_text)[0] # get text embedding from CLAP model
            dissimilarity_score = F.cosine_similarity(generated_audio_embedding, text_embedding.repeat(generated_audio_embedding.shape[0], 1), dim=-1)
            scores.append({
                "scene_text": scene_text, "dissimilarity_score": dissimilarity_score.cpu().numpy().tolist()[0]
            })
    with open(save_path, 'w', encoding='utf-8') as f:        
        json.dump(scores, f, indent=4)
        print(f"Saved random generation scores to {save_path}")
    return scores


def scores_audio_conditioner_generation(path_to_chunks, path_to_ac_weights="./saves/model_checkpoint.pt", save_path="./report/ac_generation_scores.json"):
    """Evaluate CLAP cosine similarity using the trained AudioConditioner pipeline."""
    audio_gen_model = StableAudioModel(num_inference_steps=50, num_waveforms_per_prompt=1, seed=42)
    descriptor = TwoDeepDescriptor(clap_dim=512, backbone_dim=256)
    descriptor.load_state_dict(torch.load(path_to_ac_weights)) # Load the trained weights of the descriptor
    descriptor.eval() # Set the descriptor to evaluation mode
    blip_model = BLIPModel()
    clap_model = CLAPModel()

    conditioner = AudioConditioner(audio_gen_model, descriptor, blip_model, clap_model)
    conditioner.eval()

    scores = []
    for chunk_file in tqdm(os.listdir(path_to_chunks)[:100], desc="Benchmarking AudioConditioner generation"):
        with open(os.path.join(path_to_chunks, chunk_file), 'r', encoding='utf-8') as in_f:
            scene_text = in_f.read()
            _, _, dissimilarity_score = conditioner(scene_text, input_type="text", audio_end_in_s=30, num_waveforms_per_prompt=1, num_inference_steps=50)
            scores.append({
                "scene_text": scene_text,
                "dissimilarity_score": dissimilarity_score.cpu().numpy().tolist()[0]
            })
    with open(save_path, 'w', encoding='utf-8') as f:        
        json.dump(scores, f, indent=4)
        print(f"Saved AudioConditioner generation scores to {save_path}")
    return scores


def plot_score_distributions(llm_scores_path="./report/llm_generation_scores.json", random_scores_path="./report/random_generation_scores.json", ac_scores_path="./report/ac_generation_scores.json", save_path="./report/score_distributions.png"):
    """Plot histograms comparing cosine similarity distributions across generation methods."""
    with open(llm_scores_path, 'r', encoding='utf-8') as f:
        llm_scores = json.load(f)
    with open(random_scores_path, 'r', encoding='utf-8') as f:
        random_scores = json.load(f)
    with open(ac_scores_path, 'r', encoding='utf-8') as f:
        ac_scores = json.load(f)

    llm_dissimilarity_scores = [entry["dissimilarity_score"] for entry in llm_scores]
    random_dissimilarity_scores = [entry["dissimilarity_score"] for entry in random_scores]
    ac_dissimilarity_scores = [entry["dissimilarity_score"] for entry in ac_scores]

    plt.figure(figsize=(12, 6))
    plt.hist(llm_dissimilarity_scores, bins=20, alpha=0.5, label='LLM Generation')
    plt.hist(random_dissimilarity_scores, bins=20, alpha=0.5, label='Random Generation')
    plt.hist(ac_dissimilarity_scores, bins=20, alpha=0.5, label='AudioConditioner Generation')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cosine Similarity Scores for Different Generation Methods')
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved score distribution plot to {save_path}")

    print(f"LLM Generation - Mean Cosine Similarity Score: {sum(llm_dissimilarity_scores)/len(llm_dissimilarity_scores):.4f}")
    print(f"Random Generation - Mean Cosine Similarity Score: {sum(random_dissimilarity_scores)/len(random_dissimilarity_scores):.4f}")
    print(f"AudioConditioner Generation - Mean Cosine Similarity Score: {sum(ac_dissimilarity_scores)/len(ac_dissimilarity_scores):.4f}")


