import argparse
import copy
import json
import math
import os
import torch
from clearml import Task, Logger, Dataset
from datetime import datetime
from timm.utils import ModelEmaV2
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm

from dataset import create_data_loader
from dl_models import timm_models_factory
from optimizers import torch_optimizer_factory
from utils import set_all_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clearml-project', type=str, help='Name of ClearML project', default=None)
    parser.add_argument('--clearml-task', type=str, help='Name of ClearML task', default=None)
    parser.add_argument('--train-ds-name', type=str, help='Name of ClearML training dataset', default=None)
    parser.add_argument('--train-ds-ver', type=str, help='Version of ClearML training dataset', default=None)
    parser.add_argument('--val-ds-name', type=str, help='Name of ClearML validation dataset', default=None)
    parser.add_argument('--val-ds-ver', type=str, help='Version of ClearML validation dataset', default=None)
    
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


def train(
    model,
    device,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    update_scheduler_by_iter,
    epochs,
    patience=None,
    ckpt_path=None
):
    best_acc = float('-inf')
    best_train_acc = float('-inf')
    
    Logger.current_logger().report_scalar(
        "accuracy", "val", iteration=0, value=0.0
    )
    Logger.current_logger().report_scalar(
        "EMA accuracy", "val", iteration=0, value=0.0
    )
    Logger.current_logger().report_scalar(
        "accuracy", "train", iteration=0, value=0.0
    )
    Logger.current_logger().report_scalar(
        "EMA accuracy", "train", iteration=0, value=0.0
    )
    
    model_ema = ModelEmaV2(model=model, device=device, decay=0.998)
    
    cnt_not_improve = 0
    
    best_model = None
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train epoch {epoch + 1}/{epochs}")
        total = 0
        correct = 0
        correct_ema = 0
        for i, (images, targets) in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            outputs_ema = model_ema.module(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            model_ema.update(model)

            _, predicted = torch.max(outputs.data, 1)
            _, predicted_ema = torch.max(outputs_ema.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            correct_ema += (predicted_ema == targets).sum().item()
            
            lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix(loss=loss.item(), train_acc=correct / total, ema_train_acc=correct_ema / total)
            Logger.current_logger().report_scalar(
                "loss", "train", iteration=(epoch * len(train_loader) + i), value=loss.item()
            )
            Logger.current_logger().report_scalar(
                "learning_rate", "train", iteration=(epoch * len(train_loader) + i), value=lr
            )
            
            if update_scheduler_by_iter:  # Update scheduler after each iteration
                scheduler.step()

        train_acc = correct / total
        train_acc_ema = correct_ema / total
        Logger.current_logger().report_scalar(
            "accuracy", "train", iteration=epoch + 1, value=train_acc
        )
        Logger.current_logger().report_scalar(
            "EMA accuracy", "train", iteration=epoch + 1, value=train_acc_ema
        )
        
        model.eval()
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val epoch {epoch + 1}/{epochs}")
        total = 0
        correct = 0
        correct_ema = 0
        with torch.no_grad():
            for i, (images, targets) in pbar:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                outputs_ema = model_ema.module(images)
                
                scores, predicted = torch.max(outputs.data, 1)
                scores_ema, predicted_ema = torch.max(outputs_ema.data, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                correct_ema += (predicted_ema == targets).sum().item()
                pbar.set_postfix(val_acc=correct / total, val_acc_ema=correct_ema / total)

        acc = correct / total
        acc_ema = correct_ema / total
        Logger.current_logger().report_scalar(
            "accuracy", "val", iteration=epoch + 1, value=acc
        )
        Logger.current_logger().report_scalar(
            "EMA accuracy", "val", iteration=epoch + 1, value=acc_ema
        )
        
        cnt_not_improve += 1
        if acc > best_acc:
            print(f"Val accuracy improved: {best_acc:.4f} ===> {acc:.4f}")
            best_acc = acc
            best_train_acc = train_acc
            best_model = copy.deepcopy(model)
            if ckpt_path:
                model_path = os.path.join(ckpt_path, f"epoch_{epoch + 1}_val_acc_{round(acc, 2)}.pth")
                torch.save(model.state_dict(), model_path)
            cnt_not_improve = 0
        
        elif acc_ema > best_acc:
            print(f"EMA val accuracy improved: {best_acc:.4f} ===> {acc_ema:.4f}")
            best_acc = acc_ema
            best_train_acc = train_acc_ema
            best_model = copy.deepcopy(model)
            if ckpt_path:
                model_path = os.path.join(ckpt_path, f"epoch_{epoch + 1}_ema_val_acc_{round(acc, 2)}.pth")
                torch.save(model.state_dict(), model_path)
            cnt_not_improve = 0
        
        elif math.isclose(acc, best_acc, abs_tol=1e-6) and best_train_acc < train_acc:
            print(f"Train accuracy improved: {best_train_acc:.4f} ===> {train_acc:.4f}")
            best_train_acc = train_acc
            best_model = copy.deepcopy(model)
            if ckpt_path:
                model_path = os.path.join(ckpt_path, f"epoch_{epoch + 1}_train_acc_{round(train_acc, 2)}.pth")
                torch.save(model.state_dict(), model_path)
            cnt_not_improve = 0
            
        if patience and cnt_not_improve >= patience:
            print(f"Early stopping after {patience} of not improving!")
            break
        
        if not update_scheduler_by_iter:  # Update scheduler after each epoch
            scheduler.step(metrics=acc)
            
    if ckpt_path:
        last_model_path = os.path.join(ckpt_path, f"last.pth")
        torch.save(model.state_dict(), last_model_path)
        
    return best_model, best_acc, best_train_acc


if __name__ == '__main__':
    args = parse_args()

    if args.seed:
        set_all_seeds(args.seed)
    
    Task.force_requirements_env_freeze(requirements_file="requirements.txt")
    task = Task.init(project_name=args.clearml_project, task_name=args.clearml_task)
    task.execute_remotely(queue_name='a100-vision')
    task.set_base_docker(
        docker_image='nvidia/cuda:11.6.2-runtime-ubuntu20.04',
        docker_arguments='--shm-size 16G',
        docker_setup_bash_script=["apt-get install -y curl ffmpeg"]
    )
    
    train_data_args = dict(
        dataset_name=args.train_ds_name,
        dataset_project=args.clearml_project,
        dataset_version=args.train_ds_ver
    )
    train_data_path = Dataset.get(**train_data_args).get_local_copy()
    
    val_data_args = dict(
        dataset_name=args.val_ds_name,
        dataset_project=args.clearml_project,
        dataset_version=args.val_ds_ver
    )
    val_data_path = Dataset.get(**val_data_args).get_local_copy()
    
    train_loader, n_classes = create_data_loader(
        image_root=train_data_path,
        imgsz=args.input_size,
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        aug_type=args.aug_strength,
        class_sample_ratio=args.cls_sample_ratio,
        batch_size=args.batch_size,
        num_worker=args.n_worker,
        mode='train',
        return_classes=False,
        random_seed=args.seed
    )
    
    val_loader, _ = create_data_loader(
        image_root=train_data_path,
        imgsz=args.input_size,
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        aug_type=args.aug_strength,
        class_sample_ratio=args.cls_sample_ratio,
        batch_size=args.batch_size,
        num_worker=args.n_worker,
        mode='val',
        return_classes=False,
        random_seed=args.seed
    )
    
    use_gpu = torch.cuda.is_available() and args.device != -1
    device = torch.device(f"cuda:{args.device}" if use_gpu else "cpu")
    
    model = timm_models_factory(
        arch=args.arch,
        n_classes=n_classes,
        device=device,
        phase='train',
        freeze_bottom=args.freeze_bottom,
        dropout=args.dropout,
        ckpt_path=None
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    optimizer = torch_optimizer_factory(optim_name=args.optimizer, lr=args.learning_rate, parameters=model.parameters())
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=24000, eta_min=1.0e-10)
    
    date_str = datetime.now().strftime('%Y%m%d-%H%M')
    if args.ckpt_path is not None:
        ckpt_path = os.path.join(args.ckpt_path, date_str)
        os.makedirs(ckpt_path, exist_ok=True)
    else:
        ckpt_path = None
        
    config_to_save = {
        'arch': args.arch,
        'n_classes': n_classes,
        'imgsz': args.imgsz,
        'mean': args.normalize_mean,
        'std': args.normalize_std
    }
    json.dump(config_to_save, open(os.path.join(ckpt_path, 'model_config.json'), 'w'))
    
    best_model, best_acc, best_train_acc = train(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        update_scheduler_by_iter=True,
        epochs=args.epochs,
        patience=args.patience,
        ckpt_path=ckpt_path
    )
