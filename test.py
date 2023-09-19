import argparse
import cv2
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from load_data.dataset import ImageDataset
from deep_learning import timm_models_factory
from utils import binary_classify_image, compute_stats, plot_binary_pr_curve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clearml-project', type=str, help='Name of ClearML project', default=None)
    parser.add_argument('--clearml-task', type=str, help='Name of ClearML task', default=None)
    parser.add_argument('--test-ds-name', type=str, help='Name of ClearML training dataset', default=None)
    parser.add_argument('--test-ds-ver', type=str, help='Version of ClearML training dataset', default=None)

    parser.add_argument(
        "--model_config_path",
        type=str,
        help="Model config path"
    )
    parser.add_argument(
        "--normalize_mean",
        nargs="*",
        type=float,
        help="Mean for input normalization",
        default=None,
    )
    parser.add_argument(
        "--normalize_std",
        nargs="*",
        type=float,
        help="Standard deviation for input normalization",
        default=None,
    )
    parser.add_argument(
        "--aug_strength",
        type=str,
        help="Augmentation strength",
        default='weak',
    )
    parser.add_argument(
        "--cls_sample_ratio",
        nargs="*",
        type=int,
        help="Sample ratio of classes",
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='adamw',
        help="Name of optimizer.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout of model"
    )
    parser.add_argument(
        "--freeze_bottom",
        action="store_true",
        help="Freeze bottom or not (finetune only top layer)"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Timm model architecture",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="CE Loss smoothing",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Number of not improving epoch to early stop",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    parser.add_argument('--ckpt-path', type=str, help='Path to save training checkpoints', default='./weights/')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--n-worker', type=int, default=-1)
    args = parser.parse_args()

    return args


def test_binary_classification(
    model_config_path,
    image_root,
    image_mode='RGB',
    pos_label_idx=1,
    threshold=None,
    model_ckpt_path=None,
    annot_file=None,
    device_id=-1
):
    test_data = ImageDataset(
        image_root=image_root,
        annotation_file=annot_file,
        return_image=False
    )
    literal_labels = test_data.get_literal_labels()

    device = "cuda:{}".format(device_id) if torch.cuda.is_available() and device_id != -1 else "cpu"

    config = json.load(open(model_config_path, 'r'))
    model = timm_models_factory(
        arch=config['arch'],
        n_classes=config['n_classes'],
        device=device,
        phase='val',
        ckpt_path=model_ckpt_path
    )

    y_true = []
    y_pred = []
    positive_probs = []
    results = []
    for path, gt in tqdm(test_data, total=len(test_data)):
        image = cv2.imread(path)

        label, logit = binary_classify_image(
            image=image,
            model=model,
            device=device,
            image_mode=image_mode,
            imgsz=config['imgsz'],
            mean=config['mean'],
            std=config['std'],
            threshold=threshold,
            pos_label_idx=pos_label_idx
        )

        results.append((path, label))
        y_true.append(gt)
        y_pred.append(label)
        positive_probs.append(logit[0][pos_label_idx])

    try:
        compute_stats(y_true, y_pred, literal_labels, mode='binary',
                      pos_label_idx=pos_label_idx, cm_plot_name='cm_ckpt.jpg')
        plt.cla()
        plt.clf()
        plot_binary_pr_curve(y_true, positive_probs)
    except ValueError:
        print("Can\'t calculate stats")

    df = pd.DataFrame(results, columns=['path', 'label'])
    print(df['label'].value_counts())
    df.to_csv('results-binary.csv')


if __name__ == '__main__':
    test_binary_classification()
