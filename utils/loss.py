import sys
import os
# Add parent directory to Python path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch.nn as nn
import torch
from utils.music_descriptor import MusicDescriptor
from utils.teaching_utils import MOOD_LIST, INSTRUMENTATION_LIST, RHYTHM_STYLE_LIST, STRUCTURE_LIST, PRODUCTION_STYLE_LIST, DYNAMICS_PROFILE_LIST, KEY_MODE_LIST, TEMPO_RANGE, DURATION_RANGE

class MSEMusicDescriptorLoss(nn.Module):
    """Simple MSE loss for music descriptors. This can be extended to include weighting for different attributes or to handle missing attributes more gracefully.
    """
    def __init__(self):
        super(MSEMusicDescriptorLoss, self).__init__()

    def forward(self, output, target):
        # Here we can compute the loss between the output and target music descriptors
        loss = 0.0
        for key in target:
            if key in output and output[key] is not None and target[key] is not None:
                loss += nn.functional.mse_loss(output[key], target[key], reduction='mean')
        return loss
