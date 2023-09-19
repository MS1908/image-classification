import argparse
import shutil
import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

from dataset import is_image_folder

IMAGE_EXTS = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ds-path',
        type=str,
        help='Path of dataset',
        default=None
    )
    parser.add_argument(
        '--ds-annot',
        type=str,
        help='Path of dataset annotation file',
        default=None
    )
    parser.add_argument(
        '--train-val-split',
        type=float,
        help='Training-validation split',
        default=None
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to use in randomization processes when splitting dataset",
    )
    args = parser.parse_args()

    if args.ds_path:
        image_paths = []
        for image_ext in IMAGE_EXTS:
            image_paths.extend(glob(args.ds_path + f'/**/*.{image_ext}', recursive=True))

        train_paths, val_paths = train_test_split(
            image_paths,
            test_size=args.train_val_split,
            random_state=args.seed
        )

        train_root = os.path.join(args.ds_path, 'train')
        val_root = os.path.join(args.ds_path, 'val')
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(val_root, exist_ok=True)
        if is_image_folder(args.ds_path):
            for p in train_paths:
                lbl = os.path.dirname(p).split('/')[-1]
                os.makedirs(os.path.join(train_root, lbl), exist_ok=True)
                shutil.move(p, os.path.join(train_root, lbl))

            for p in val_paths:
                lbl = os.path.dirname(p).split('/')[-1]
                os.makedirs(os.path.join(val_root, lbl), exist_ok=True)
                shutil.move(p, os.path.join(val_root, lbl))

        else:
            for p in train_paths:
                shutil.move(p, train_root)

            for p in val_paths:
                shutil.move(p, val_root)

    if args.ds_annot:
        df = pd.read_csv(args.ds_annot)
        train_df, val_df = train_test_split(
            df,
            test_size=args.train_val_split,
            random_state=args.seed
        )
        d = os.path.dirname(args.ds_annot)
        train_df.to_csv(os.path.join(d, 'train.csv'))
        val_df.to_csv(os.path.join(d, 'val.csv'))
        os.remove(args.ds_annot)
