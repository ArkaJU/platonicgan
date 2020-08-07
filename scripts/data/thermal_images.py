from torch.utils.data import Dataset
import os
import cv2
from scripts.utils.utils import normalize, convert_to_int
import scripts.utils.io as dh
import numpy as np


class CustomDataset(Dataset):

    def __init__(self, param, mode):
        """
        Args:
            path_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.param = param
        self.path_dir = os.path.join('/content/drive/My Drive', param.data.path_dir, mode)

        self.cube_len = param.data.cube_len
        self.subject_paths = sorted([name for name in os.listdir(self.path_dir)])
        #print(self.subject_paths)
        self.dataset_length = len(self.subject_paths)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        #print(f'CustomDataset: {idx}')
        subject_path = os.path.join(self.path_dir, self.subject_paths[idx])

        views = [ 'Right_Lateral.mat',
                  'Right_Oblique.mat',
                  'Frontal.mat',
                  'Left_Oblique.mat',
                  'Left_Lateral.mat' ]
        images = []
        for image_path in views:
          
          full_image_path = os.path.join(subject_path, image_path)
          image = dh.read_image(full_image_path, self.cube_len)
          
          if self.param.renderer.type == 'visual_hull':
            image = image[[3]]
          elif self.param.renderer.type == 'absorption_only':
            image = image[[0]]
          elif self.param.renderer.type == 'emission_absorption':
            image[0:3] = image[0:3] * image[3]
          
          images.append(image)

        #stacking grayscale images
        images = np.concatenate(images, axis=0)
        #print(f"images.shape: {images.shape}")
        return images, idx
