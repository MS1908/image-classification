import argparse
from clearml import Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-name', type=str, default=None)
    parser.add_argument('--train-path', type=str, default=None)
    parser.add_argument('--val-path', type=str, default=None)
    parser.add_argument('--train-name', type=str, default=None)
    parser.add_argument('--val-name', type=str, default=None)
    args = parser.parse_args()
    
    if (args.train_path is not None) and (args.train_name is not None):
        train_dataset = Dataset.create(
            dataset_name=args.train_name,
            dataset_project=args.project_name
        )
        train_dataset.add_files(path=args.train_path)
        train_dataset.upload()
        train_dataset.finalize()
    
    if (args.val_path is not None) and (args.val_name is not None):
        val_dataset = Dataset.create(
            dataset_name=args.val_name,
            dataset_project=args.project_name
        )
        val_dataset.add_files(path=args.val_path)
        val_dataset.upload()
        val_dataset.finalize()
