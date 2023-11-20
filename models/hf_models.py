from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification


def hf_models_factory(
    model_name_or_path,
    labels,
    trust_remote_code=False,
    use_hf_pretrain=False,
    build_img_processor=False,
    ignore_mismatched_sizes=True
):
    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in enumerate(labels)}
    
    if use_hf_pretrain:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
            trust_remote_code=trust_remote_code
        )

        model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            trust_remote_code=trust_remote_code,
        )

    else:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            trust_remote_code=trust_remote_code
        )

        model = AutoModelForImageClassification.from_config(config)

    if build_img_processor:
        processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )
    else:
        processor = None

    return config, processor, model


if __name__ == '__main__':
    config, processor, model = hf_models_factory(
        model_name_or_path='microsoft/swin-tiny-patch4-window7-224',
        labels=['a', 'b'],
        use_hf_pretrain=True
    )
