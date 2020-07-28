import torch
from scripts.renderer.absorption_only import AbsorptionOnlyRenderer


class VisualHullRenderer():
    def __init__(self, param):
        self.param = param

    def render(self, volume, axis=2):
        print(f"inside VH: {volume.shape}")
        print(f"volume: {volume.device}")
        image = torch.sum(volume, dim=axis)
        image = torch.ones_like(image) - torch.exp(-image)
        print(f"inside VH: {image.shape}")
        print(f"image: {image.device}")
        return image
