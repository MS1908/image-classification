import random
import torch
import numpy as np
from torch.utils import data

from load_data.dataset import CustomSampleImageDataset
from utils import image_augmentation_pipeline


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loader(
    image_root,
    annotation_file=None,
    imgsz=None,
    normalize_mean=None,
    normalize_std=None,
    aug_type='weak',
    class_sample_ratio=None,
    batch_size=32,
    num_worker=8,
    mode='train',
    return_classes=False,
    collate_fn=None,
    random_seed=None
):
    pipeline = image_augmentation_pipeline(
        augmentation_type=aug_type,
        imgsz=imgsz,
        phase=mode,
        mean=normalize_mean,
        std=normalize_std
    )

    if mode == 'train':
        dataset = CustomSampleImageDataset(
            image_root,
            annotation_file,
            transforms_pipeline=pipeline,
            class_sampling_ratio=class_sample_ratio
        )

    else:  # Don't oversampling val dataset
        dataset = CustomSampleImageDataset(
            image_root,
            annotation_file,
            transforms_pipeline=pipeline,
            class_sampling_ratio=None
        )

    if random_seed:
        g = torch.Generator()
        g.manual_seed(random_seed)
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=mode == 'train',
            num_workers=num_worker,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=collate_fn
        )
    else:
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=mode == 'train',
            num_workers=num_worker,
            collate_fn=collate_fn
        )
    if return_classes:
        return data_loader, dataset.n_classes, dataset.get_literal_labels()
    else:
        return data_loader, dataset.n_classes
