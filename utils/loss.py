import torch.nn as nn
from utils.music_descriptor import REGRESSION_ATTRIBUTES, CLASSIFICATION_ATTRIBUTES

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


class AdaptedMusicDescriptorLoss(nn.Module):
    """An adapted loss function that can handle missing attributes and apply different weights to different attributes based on their importance.
    """
    def __init__(self, attribute_weights=None):
        super(AdaptedMusicDescriptorLoss, self).__init__()
        self.attribute_weights = attribute_weights if attribute_weights is not None else {}

    def forward(self, output, target):
        loss = 0.0
        for key in target:
            if key in output and output[key] is not None and target[key] is not None:
                if key in CLASSIFICATION_ATTRIBUTES:
                    # If the attribute is a classification attribute, we use cross-entropy loss
                    loss += self.attribute_weights.get(key, 1.0) * nn.functional.cross_entropy(output[key], target[key], reduction='mean') 
                elif key in REGRESSION_ATTRIBUTES:
                    # If the attribute is a regression attribute, we use MSE loss
                    loss += self.attribute_weights.get(key, 1.0) * nn.functional.mse_loss(output[key], target[key], reduction='mean')
        return loss