from torchvision import datasets
from torch.utils import data


def torch_dataset_factory(dataset_name, dataset_root, bs=32):
    if dataset_name.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(root=dataset_root,
                                         train=True,
                                         download=True)
        test_dataset = datasets.CIFAR10(root=dataset_root,
                                        train=False,
                                        download=True)

    elif dataset_name.lower() == 'cifar100':
        train_dataset = datasets.CIFAR100(root=dataset_root,
                                          train=True,
                                          download=True)
        test_dataset = datasets.CIFAR100(root=dataset_root,
                                         train=False,
                                         download=True)

    else:
        raise NotImplementedError(f"Image classification for dataset {dataset_name} is not impldmented")

    train_loader = data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2)
    n_classes = len(train_dataset.classes)
    return train_loader, test_loader, n_classes
