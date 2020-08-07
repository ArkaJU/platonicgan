from scripts.data import image, caltech_ucsd_birds, thermal_images

dataset_dict = {
    'image': image.ImageDataset,
    'ucb':  caltech_ucsd_birds.UCBDataset,
    'custom': thermal_images.CustomDataset
}
