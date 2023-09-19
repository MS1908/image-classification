import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def image_preprocess_torch(
    image,
    to_rgb=False,
    imgsz=None,
    mean=None,
    std=None,
    squeeze=False
):
    """Convert BGR OpenCV image to PyTorch tensor (to use as input for PyTorch model)

    Args:
        image (np.ndarray): An BGR OpenCV image
        to_rgb (bool, optional): Convert from BGR color space to RGB flag. Defaults to False.
        imgsz (list[int] / tuple[int], optional): Desired image size to resize the image. Defaults to None.
        mean (list[float] / tuple[float], optional): Mean to use in image normalization. Defaults to None.
        std (list[float] / tuple[float], optional): Standard deviation to use in image normalization. Defaults to None.
        squeeze (bool, optional): If set to False, then return the tensor, else add a batch size dimension
        to the output tensor. Defaults to False.
    """
    if to_rgb:
        ret_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    else:
        ret_image = image.copy()
        
    if isinstance(imgsz, tuple) or isinstance(imgsz, list):
        assert len(imgsz) <= 2, "List/tuple of sizes must be of format (h, w) or (imgsz,)"
        
        if len(imgsz) == 2:
            output_h, output_w = imgsz
            resize_op = A.Resize(height=output_h, width=output_w)
        else:
            # Only longest size is provided --> Use center padding to resize the image to desired size
            resize_op = A.Compose([
                A.LongestMaxSize(max_size=imgsz[0]),
                A.PadIfNeeded(min_height=imgsz[0], min_width=imgsz[0],
                              border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
            ])
    elif imgsz:
        # Only longest size is provided
        resize_op = A.Compose([
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz,
                          border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        ])
    else:
        resize_op = None
        
    if mean and std:
        normalize_op = A.Normalize(mean=mean, std=std)
    else:
        normalize_op = None
        
    pipeline = []
    if resize_op:
        pipeline.append(resize_op)
    if normalize_op:
        pipeline.append(normalize_op)
    pipeline.append(ToTensorV2())  # Convert to PyTorch tensor
    
    transforms = A.Compose(pipeline, p=1.0)
    ret_image = transforms(image=ret_image)['image']
    
    if not squeeze:
        ret_image = ret_image.unsqueeze(0)  # Add batch size dimension
        
    return ret_image


def image_preprocess_np(
    image,
    to_rgb=False,
    to_chw_order=False,
    imgsz=None,
    mean=None,
    std=None,
    squeeze=False
):
    """Preprocess BGR OpenCV image, and keep it as numpy array (to use as input for ONNX model)

    Args:
        image (np.ndarray): An BRG OpenCV image
        to_rgb (bool, optional): Convert from BGR color space to RGB flag. Defaults to False.
        to_chw_order (bool, optional): Convert image from HWC order to CHW order, to make it compatible
        with model trained using PyTorch framework. Defaults to False.
        imgsz (list[int] / tuple[int], optional): Desired image size to resize the image. Defaults to None.
        mean (list[float] / tuple[float], optional): Mean to use in image normalization. Defaults to None.
        std (list[float] / tuple[float], optional): Standard deviation to use in image normalization. Defaults to None.
        squeeze (bool, optional): If set to False, then return the tensor, else add a batch size dimension
        to the output tensor. Defaults to False.
    """
    if to_rgb:
        ret_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    else:
        ret_image = image.copy()

    if isinstance(imgsz, tuple) or isinstance(imgsz, list):
        assert len(imgsz) <= 2, "List/tuple of sizes must be of format (h, w) or (imgsz,)"

        if len(imgsz) == 2:
            output_h, output_w = imgsz
            resize_op = A.Resize(height=output_h, width=output_w)
        else:
            # Only longest size is provided --> Use center padding to resize the image to desired size
            resize_op = A.Compose([
                A.LongestMaxSize(max_size=imgsz[0]),
                A.PadIfNeeded(min_height=imgsz[0], min_width=imgsz[0],
                              border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
            ])
    elif imgsz:
        # Only longest size is provided
        resize_op = A.Compose([
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz,
                          border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        ])
    else:
        resize_op = None

    if mean and std:
        normalize_op = A.Normalize(mean=mean, std=std)
    else:
        normalize_op = None

    pipeline = []
    if resize_op:
        pipeline.append(resize_op)
    if normalize_op:
        pipeline.append(normalize_op)

    transforms = A.Compose(pipeline, p=1.0)
    ret_image = transforms(image=ret_image)['image']

    if to_chw_order:
        ret_image = np.transpose(ret_image, (2, 0, 1))  # Convert image from HWC order to CHW order
    if not squeeze:
        ret_image = np.expand_dims(ret_image, axis=0)  # Add batch size dimension

    return ret_image


def image_augmentation_pipeline(augmentation_type='weak', phase='train',
                                imgsz=None, mean=None, std=None):
    """Pipeline for image augmentation

    Args:
        augmentation_type (str, optional): Augmentation strength (either 'strong' or 'weak'). Defaults to 'weak'.
        phase (str, optional): The mode of augmentation pipeline. Either in 'train' mode (for training) or 'val' mode (for evaluation).
        Defaults to 'train'. Defaults to 'train'.
        imgsz (list, optional): Target sizes for image. Defaults to None.
        mean (list, optional): Normalize mean. Defaults to None.
        std (list, optional): Normalize standard deviation. Defaults to None.

    Returns:
        list: A list of augmentation operations (from albumentations library)
    """

    if isinstance(imgsz, tuple) or isinstance(imgsz, list):
        assert len(imgsz) <= 2, "List/tuple of sizes must be of format (h, w) or (imgsz,)"

        if len(imgsz) == 2:
            output_h, output_w = imgsz
            resize_op = A.Resize(height=output_h, width=output_w)
        else:
            # Only longest size is provided --> Use center padding to resize the image to desired size
            resize_op = A.Compose([
                A.LongestMaxSize(max_size=imgsz[0]),
                A.PadIfNeeded(min_height=imgsz[0], min_width=imgsz[0],
                              border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
            ])
    elif imgsz:
        # Only longest size is provided
        resize_op = A.Compose([
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz,
                          border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        ])
    else:
        resize_op = None

    if mean and std:
        normalize_op = A.Normalize(mean=mean, std=std)
    else:
        normalize_op = None

    if phase != 'train' or augmentation_type not in ['weak', 'strong']:
        pipeline = []

    elif augmentation_type == 'weak':  # Weak data augmentation (just blur, adjust brightness, rotation with small angle)
            pipeline = [
                A.GaussNoise(p=0.2),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1)
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast()
                ], p=0.3),
                A.HueSaturationValue(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.025,
                    rotate_limit=10,
                    shift_limit_x=0.025,
                    shift_limit_y=0.025,
                    p=0.3)
            ]

    else:  # Strong data augmentation (noise, blur, optical distortion, adjust brightness, rotation with big angle)
        pipeline = [
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.5),
                A.ElasticTransform(),
                A.GridDistortion()
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1)
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast()
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.025,
                rotate_limit=90,
                shift_limit_x=0.025,
                shift_limit_y=0.025,
                p=0.3),
        ]

    if resize_op:
        pipeline.append(resize_op)
    if normalize_op:
        pipeline.append(normalize_op)
    pipeline.append(ToTensorV2())  # Convert to PyTorch tensor

    return pipeline
