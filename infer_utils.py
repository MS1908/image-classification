import torch
import numpy as np
import torch.nn.functional as F

from image_processing_utils import image_preprocess_torch


def binary_classify_image(
    image,
    model,
    device,
    image_mode='RGB',
    imgsz=None,
    mean=None,
    std=None,
    threshold=None,
    pos_label_idx=1
):
    """
    Binary classification of an image, using PyTorch model

    Args:
        image: An BGR OpenCV image
        model: PyTorch model use for image classification
        device: Device to infer on
        image_mode: The desired image mode for model. Default to RGB
        imgsz: The desired image size for model. If None then no resize is performed.
        mean: Mean for normalization. If None then no normalization is performed.
        std: Standard deviation for normalization. If None then no normalization is performed.
        threshold: Positive prediction threshold
        pos_label_idx: Index of positive label (In some problem, label 0 is positive)

    Returns:
        Predicted label and logits.
    """

    input_image = image_preprocess_torch(
        image=image.copy(),  # Make sure that input image is not mutated.
        to_rgb=(image_mode == 'RGB'),
        imgsz=imgsz,
        mean=mean,
        std=std,
        squeeze=False  # PyTorch model need batch size dimension
    )

    if device is not None:
        # Load model and image on device for inference
        input_image = input_image.to(device)
        model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.forward(input_image)
        logit = F.softmax(outputs, dim=-1).cpu().numpy()

    if threshold is None:
        label = np.argmax(logit)
    else:
        # binary classification, hence indices of labels are only 0 and 1
        label = pos_label_idx if logit[0][pos_label_idx] >= threshold else 1 - pos_label_idx

    return label, logit


def classify_image(
    image,
    model,
    device,
    image_mode='RGB',
    imgsz=None,
    mean=None,
    std=None
):
    """
    Image classification using PyTorch model

    Args:
        image: An BGR OpenCV image
        model: PyTorch model use for image classification
        device: Device to infer on
        image_mode: The desired image mode for model. Default to RGB
        imgsz: The desired image size for model. If None then no resize is performed.
        mean: Mean for normalization. If None then no normalization is performed.
        std: Standard deviation for normalization. If None then no normalization is performed.

    Returns:
        Predicted label and logits.
    """

    input_image = image_preprocess_torch(
        image=image.copy(),  # Make sure that input image is not mutated.
        to_rgb=(image_mode == 'RGB'),
        imgsz=imgsz,
        mean=mean,
        std=std,
        squeeze=False  # PyTorch model need batch size dimension
    )

    if device is not None:
        # Load model and image on device for inference
        input_image = input_image.to(device)
        model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.forward(input_image)
        logit = F.softmax(outputs, dim=-1).cpu().numpy()

    label = np.argmax(logit)

    return label, logit
