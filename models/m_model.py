import sys
import os
# Add parent directory to Python path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union
import random
from utils.music_descriptor import ATTRIBUTES_THAT_ARE_LISTS, CLASSIFICATION_ATTRIBUTES, REGRESSION_ATTRIBUTES, MusicDescriptor
from utils.teaching_utils import MOOD_LIST, INSTRUMENTATION_LIST, RHYTHM_STYLE_LIST, STRUCTURE_LIST, PRODUCTION_STYLE_LIST, DYNAMICS_PROFILE_LIST, TEMPO_RANGE, DURATION_RANGE, KEY_MODE_LIST

class m_model(nn.Module):
    
    """
    A multi-task model that takes a scene description as input and outputs a MusicDescriptor JSON object. The model consists of a shared backbone for feature extraction and multiple heads for predicting different attributes of the music descriptor. The generate_music_descriptor method combines the outputs of the heads into a structured MusicDescriptor object, applying any necessary post-processing (e.g., mapping class indices to labels, applying top-p sampling for multi-label outputs).

    Forward pass outputs:
    - mood: A list of strings describing the mood of the music (e.g., "dark", "epic", "melancholic").
    - energy: A float between 0 and 1 indicating the energy level of the music (0 = very calm, 1 = very energetic).
    - valence: A float between 0 and 1 indicating the valence of the music (0 = very negative/sad, 1 = very positive/happy).
    - tempo: An integer representing the tempo of the music in beats per minute (BPM).
    - key_mode: A string indicating the musical key and mode (e.g., "major", "minor", "ambiguous").
    - harmonic_tension: A float between 0 and 1 indicating the harmonic tension of the music (0 = very consonant, 1 = very dissonant).
    - instrumentation: A list of strings describing the instruments used in the music (e.g., "low strings", "brass", "war drums"...).
    - rhythm_style: A string describing the rhythm style of the music (e.g., "sparse", "pulsing", "syncopated"...).
    - structure: A string describing the structure of the music (e.g., "slow-build -> climax", "loop-based"...).
    - texture_density: A float between 0 and 1 indicating the density of the musical texture (0 = very sparse, 1 = very dense).
    - production_style: A list of strings describing the production style of the music (e.g., "lo-fi", "cinematic", "electronic"...).
    - dynamics_profile: A string describing the dynamics profile of the music (e.g., "gradual build", "sudden drops", "consistent energy"...).
    - duration: An integer representing the desired duration of the music in seconds.


    """

    def __init__(self,
                 backbone: nn.Module,
                 mood_classifier: nn.Module,
                 energy_regressor: nn.Module,
                 valence_regressor: nn.Module,
                 tempo_regressor: nn.Module,
                 key_mode_classifier: nn.Module,
                 harmonic_tension_regressor: nn.Module,
                 instrumentation_classifier: nn.Module,
                 rhythm_style_classifier: nn.Module,
                 structure_classifier: nn.Module,
                 texture_density_regressor: nn.Module,
                 production_style_classifier: nn.Module,
                 dynamics_profile_classifier: nn.Module,
                 duration_regressor: nn.Module,
                 top_p: float = 0.9,
                 **args):
        super(m_model, self).__init__()
        self.backbone = backbone
        self.mood_classifier = mood_classifier
        self.energy_regressor = energy_regressor
        self.valence_regressor = valence_regressor
        self.tempo_regressor = tempo_regressor
        self.key_mode_classifier = key_mode_classifier
        self.harmonic_tension_regressor = harmonic_tension_regressor
        self.instrumentation_classifier = instrumentation_classifier
        self.rhythm_style_classifier = rhythm_style_classifier
        self.structure_classifier = structure_classifier
        self.texture_density_regressor = texture_density_regressor
        self.production_style_classifier = production_style_classifier
        self.dynamics_profile_classifier = dynamics_profile_classifier
        self.duration_regressor = duration_regressor
        self.top_p = top_p

        self.attributes_that_are_lists = ATTRIBUTES_THAT_ARE_LISTS
        self.classification_attributes = CLASSIFICATION_ATTRIBUTES
        self.regression_attributes = REGRESSION_ATTRIBUTES

    def forward(self, x):
        features = self.backbone(x)
        mood = self.mood_classifier(features)
        energy = self.energy_regressor(features)
        valence = self.valence_regressor(features)
        tempo = self.tempo_regressor(features)
        key_mode = self.key_mode_classifier(features)
        harmonic_tension = self.harmonic_tension_regressor(features)
        instrumentation = self.instrumentation_classifier(features)
        rhythm_style = self.rhythm_style_classifier(features)
        structure = self.structure_classifier(features)
        texture_density = self.texture_density_regressor(features)
        production_style = self.production_style_classifier(features)
        dynamics_profile = self.dynamics_profile_classifier(features)
        duration = self.duration_regressor(features)

        output = {
            "mood": mood,
            "energy": energy,
            "valence": valence,
            "tempo": tempo,
            "key_mode": key_mode,
            "harmonic_tension": harmonic_tension,
            "instrumentation": instrumentation,
            "rhythm_style": rhythm_style,
            "structure": structure,
            "texture_density": texture_density,
            "production_style": production_style,
            "dynamics_profile": dynamics_profile,
            "duration": duration
        }

        # Post-process outputs to convert them into the expected formats (e.g., mapping class indices to labels, applying activation functions)
        # The classification heads will output logits, so we need to apply softmax to get probabilities and then map to labels. The regression heads will output raw values that may need to be scaled or clamped to the expected ranges.
        for attribute in self.classification_attributes:
            output[attribute] = F.softmax(output[attribute], dim=-1)
        for attribute in self.regression_attributes:
            output[attribute] = torch.sigmoid(output[attribute])  # Apply sigmoid to regression outputs to bound them between 0 and 1
      

        return output
    
    def to_range_int(self, value, min_val, max_val):
        """Converts a normalized value between 0 and 1 to an integer in the specified range.
        """
        return int(value * (max_val - min_val) + min_val)
    
    def generate_music_descriptor(self, x, top_p: Union[float, None] = None):
        if top_p is None:
            top_p = self.top_p
        output = self.forward(x)


        for attribute in self.attributes_that_are_lists:
            probs = output[attribute]
            indices = torch.where(probs > top_p)[1]
            labels = [globals()[f"{attribute.upper()}_LIST"][i] for i in indices]
            output[attribute] = labels

        for attribute in self.classification_attributes:
            if attribute not in self.attributes_that_are_lists:
                probs = output[attribute]
                index = torch.argmax(probs, dim=-1).item()
                label = globals()[f"{attribute.upper()}_LIST"][index]
                output[attribute] = label

        for attribute in self.regression_attributes:
            output[attribute] = output[attribute].item()  # Convert tensor to scalar value
            if attribute == "tempo":
                output[attribute] = self.to_range_int(output[attribute], *TEMPO_RANGE)
            elif attribute == "duration":
                output[attribute] = self.to_range_int(output[attribute], *DURATION_RANGE)

        # Construct the MusicDescriptor object using the post-processed outputs
        music_descriptor = MusicDescriptor(
            mood=output["mood"],
            energy=output["energy"],
            valence=output["valence"],
            tempo=output["tempo"],
            key_mode=output["key_mode"],
            harmonic_tension=output["harmonic_tension"],
            instrumentation=output["instrumentation"],
            rhythm_style=output["rhythm_style"],
            structure=output["structure"],
            texture_density=output["texture_density"],
            production_style=output["production_style"],
            dynamics_profile=output["dynamics_profile"],
            duration=output["duration"]
        )
        return music_descriptor


class one_deep_m_model(m_model):
    def __init__(self, clap_dim, backbone_dim: int, **args):
        backbone = nn.Linear(clap_dim, backbone_dim)
        mood_classifier = nn.Linear(backbone_dim, len(MOOD_LIST))
        energy_regressor = nn.Linear(backbone_dim, 1)
        valence_regressor = nn.Linear(backbone_dim, 1)
        tempo_regressor = nn.Linear(backbone_dim, 1)
        key_mode_classifier = nn.Linear(backbone_dim, len(KEY_MODE_LIST))
        harmonic_tension_regressor = nn.Linear(backbone_dim, 1)
        instrumentation_classifier = nn.Linear(backbone_dim, len(INSTRUMENTATION_LIST))
        rhythm_style_classifier = nn.Linear(backbone_dim, len(RHYTHM_STYLE_LIST))
        structure_classifier = nn.Linear(backbone_dim, len(STRUCTURE_LIST))
        texture_density_regressor = nn.Linear(backbone_dim, 1)
        production_style_classifier = nn.Linear(backbone_dim, len(PRODUCTION_STYLE_LIST))
        dynamics_profile_classifier = nn.Linear(backbone_dim, len(DYNAMICS_PROFILE_LIST))
        duration_regressor = nn.Linear(backbone_dim, 1)

        super(one_deep_m_model, self).__init__(
            backbone=backbone,
            mood_classifier=mood_classifier,
            energy_regressor=energy_regressor,
            valence_regressor=valence_regressor,
            tempo_regressor=tempo_regressor,
            key_mode_classifier=key_mode_classifier,
            harmonic_tension_regressor=harmonic_tension_regressor,
            instrumentation_classifier=instrumentation_classifier,
            rhythm_style_classifier=rhythm_style_classifier,
            structure_classifier=structure_classifier,
            texture_density_regressor=texture_density_regressor,
            production_style_classifier=production_style_classifier,
            dynamics_profile_classifier=dynamics_profile_classifier,
            duration_regressor=duration_regressor,
             **args
        )


def test():
    model = one_deep_m_model(clap_dim=512, backbone_dim=256)
    x = torch.randn(1, 512)  # Simulated CLAP features
    output = model.forward(x)
    music_descriptor = model.generate_music_descriptor(x, top_p=0.05)
    diff_tensor = music_descriptor.to_differentiable_tensor()

    for attribute, value in output.items():
        print("\n")
        print(f"{attribute}: {value}")
        print(f"{attribute} (post-processed): {getattr(music_descriptor, attribute)}")
        print(f"{attribute} (differentiable tensor): {diff_tensor[attribute]}")

    import utils.loss
    criterion = utils.loss.MSEMusicDescriptorLoss()
    print("\nLoss:", criterion(output, diff_tensor))
    print("0 Loss:", criterion(diff_tensor, diff_tensor))  # Loss between the same descriptor should be 0

if __name__ == "__main__":

        
    test()
        
