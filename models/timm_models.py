import timm
import torch


def freeze_params(arch, model, n_conv_blocks_to_train=0):
    """Freeze part of the weights for model finetuning

    Args:
        arch: Architecture of model
        model: A PyTorch model built using timm library
        n_conv_blocks_to_train: The number of convolution blocks at the top of the model
        you want to train. Defaults to 0.
    """
    for param in model.parameters():
        param.requires_grad = False

    if 'convnext' in arch:
        # Unfreeze only the head of the model
        for param in model.head.parameters():
            param.requires_grad = True
        
        # Unfreeze the top n_conv_blocks_to_train convolution blocks of the model
        if n_conv_blocks_to_train > 0:
            for param in model.stages[-n_conv_blocks_to_train:].parameters():
                param.requires_grad = True

    elif 'mobilevit' in arch:
        for param in model.head.parameters():
            param.requires_grad = True
        for param in model.final_conv.parameters():
            param.requires_grad = True
        
        if n_conv_blocks_to_train > 0:
            for param in model.stages[-n_conv_blocks_to_train:].parameters():
                param.requires_grad = True

    elif 'vit' in arch:
        for param in model.head.parameters():
            param.requires_grad = True
        for param in model.fc_norm.parameters():
            param.requires_grad = True
        for param in model.norm.parameters():
            param.requires_grad = True
        
        if n_conv_blocks_to_train > 0:
            for param in model.blocks[-n_conv_blocks_to_train:].parameters():
                param.requires_grad = True

    elif 'efficientnet' in arch:
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.global_pool.parameters():
            param.requires_grad = True
        for param in model.conv_head.parameters():
            param.requires_grad = True
        for param in model.bn2.parameters():
            param.requires_grad = True
        
        if n_conv_blocks_to_train > 0:
            for param in model.blocks[-n_conv_blocks_to_train:].parameters():
                param.requires_grad = True

    else:
        # Haven't thought of handling these architectures yet,
        # so I will simply unfreeze and train the entire model
        for param in model.parameters():
            param.requires_grad = True


def timm_models_factory(
    arch, 
    n_classes,
    device=None,
    use_pretrain=True,
    phase='train',
    freeze_bottom=False,
    n_conv_blocks_to_train=0,
    dropout=0.0, 
    ckpt_path=None
):
    """Build a model for image classification by using pytorch image model (timm) library

    Args:
        arch (str): Model architecture
        
        n_classes (int): The number of output classes. If both this argument and n_classes are specified in config,
        this argument will be given preference.
        
        device: Device to load model on. Defaults to None.

        use_pretrain (bool): Use pretrain weight from timm library or not.
        
        phase (str): The mode of model. Either in 'train' mode (for training) or 'val' mode (for evaluation). 
        Defaults to 'train'.
        
        freeze_bottom (bool): Freeze the bottom of model, only finetune the top few layers. (For timm, we freeze the bottom blocks,
        and finetune the top blocks.)

        n_conv_blocks_to_train (int): Number of convolution blocks that will be trained. Defaults to 0
        
        dropout (float): Dropout of model (when training)
        
        ckpt_path (str, optional): Checkpoint path. Defaults to None.

    Raises:
        ValueError: When phase is not either 'train' or 'val'

    Returns:
        A PyTorch model built by timm library in accordance to the config and specified arguments.
    """
    try:
        model = timm.create_model(arch,
                                  pretrained=use_pretrain,
                                  num_classes=n_classes,
                                  drop_rate=dropout)
    except RuntimeError:
        # Doesn't have ImageNet pretrain for this architecture
        model = timm.create_model(arch,
                                  pretrained=False,
                                  num_classes=n_classes,
                                  drop_rate=dropout)

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)

    if device is not None:
        model.to(device)

    if phase == 'train':
        if freeze_bottom:
            freeze_params(arch, model, n_conv_blocks_to_train)  # Freeze param for finetuning model
        model.train()
    elif phase == 'val':
        model.eval()
    else:
        raise ValueError(f"Invalid argument for 'phase'. Expect ['train', 'val'], found {phase}")

    return model
