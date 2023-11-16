import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy

from load_data import create_data_loader
from utils import set_all_seeds
from models import hf_models_factory
from training_utils import torch_optimizer_factory


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-ds-path',
        type=str,
        help='Path of training dataset',
        default=None
    )
    parser.add_argument(
        '--train-ds-annot',
        type=str,
        help='Path of training dataset annotation file',
        default=None
    )
    parser.add_argument(
        '--val-ds-path',
        type=str,
        help='Path of validation dataset',
        default=None
    )
    parser.add_argument(
        '--val-ds-annot',
        type=str,
        help='Path of validation dataset annotation file',
        default=None
    )

    parser.add_argument(
        "--input_size",
        nargs="*",
        type=int,
        help="Input size of model",
        default=[224],
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
        "--model_name_or_path",
        type=str,
        default=None,
        help="HuggingFace model path (or name)",
    )
    parser.add_argument(
        "--use_pretrain",
        action="store_true",
        help="Freeze bottom or not (finetune only top layer)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        '--n-worker',
        type=int,
        help='Number of workers to use when training',
        default=0
    )
    parser.add_argument(
        '--ckpt-path',
        type=str,
        help='Path to save training checkpoints',
        default='./weights/'
    )
    args = parser.parse_args()

    return args


class ModelModule(pl.LightningModule):

    def __init__(self, model_name_or_path, labels, id2label, label2id, use_pretrain=True, optim_name='adamw', lr=1e-3):
        super().__init__()
        config, processor, model = hf_models_factory(
            model_name_or_path=model_name_or_path,
            labels=labels,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=False,
            use_hf_pretrain=use_pretrain,
            build_img_processor=False  # Image already processed during augmentation process
        )
        self.model = model
        self.train_accuracy = Accuracy(task="multiclass", num_classes=len(labels))
        self.val_accuracy = Accuracy(task="multiclass", num_classes=len(labels))
        self.optim_name = optim_name
        self.lr = lr

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        self.log("train_loss", outputs.loss)
        self.log(
            "train_acc",
            self.train_accuracy(outputs.logits, batch['labels']),
            on_step=True,
            on_epoch=True
        )
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        self.log("val_loss", outputs.loss)
        self.log(
            "val_acc",
            self.val_accuracy(outputs.logits, batch['labels']),
            on_step=True,
            on_epoch=True
        )
        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch_optimizer_factory(self.optim_name, self.lr, self.model.parameters())
        return optimizer


if __name__ == '__main__':
    args = parse_args()

    if args.seed:
        set_all_seeds(args.seed)

    train_loader, n_classes, labels = create_data_loader(
        image_root=args.train_ds_path,
        annotation_file=args.train_ds_annot,
        imgsz=args.input_size,
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        aug_type=args.aug_strength,
        class_sample_ratio=args.cls_sample_ratio,
        batch_size=args.batch_size,
        num_worker=args.n_worker,
        mode='train',
        return_classes=True,
        collate_fn=collate_fn,
        random_seed=args.seed
    )
    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in enumerate(labels)}

    val_loader, _ = create_data_loader(
        image_root=args.val_ds_path,
        annotation_file=args.val_ds_annot,
        imgsz=args.input_size,
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        aug_type=args.aug_strength,
        class_sample_ratio=args.cls_sample_ratio,
        batch_size=args.batch_size,
        num_worker=args.n_worker,
        mode='val',
        return_classes=False,
        collate_fn=collate_fn,
        random_seed=args.seed
    )

    model_module = ModelModule(
        model_name_or_path=args.model_name_or_path,
        labels=labels,
        id2label=id2label,
        label2id=label2id,
        use_pretrain=args.use_pretrain,
        optim_name=args.optimizer,
        lr=args.learning_rate
    )
    ckpt = ModelCheckpoint(
        filename="{epoch}-{val_acc:.4f}", save_last=True, save_top_k=1, monitor="val_acc", mode="max"
    )
    trainer = pl.Trainer(
        accelerator="auto",
        precision=16,
        devices=1,
        max_epochs=args.epochs,
        callbacks=[ckpt],
        log_every_n_steps=1
    )
    trainer.fit(model_module, train_loader, val_loader)

    model_arch = args.model_name_or_path.split('/')[-1]
    os.makedirs(os.path.join(args.ckpt_path, model_arch), exist_ok=True)
    model_module.model.save_pretrained(os.path.join(args.ckpt_path, model_arch, 'best'), from_pt=True)
