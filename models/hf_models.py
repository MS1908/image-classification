from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification


def hf_models_factory(
    model_name_or_path,
    labels,
    id2label,
    label2id,
    trust_remote_code=False,
    use_hf_pretrain=False,
    ignore_mismatched_sizes=True
):
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

    return config, model


if __name__ == '__main__':
    config, model = build_hf_model(
        model_name_or_path='microsoft/swin-tiny-patch4-window7-224',
        labels=['a', 'b'],
        id2label={0: 'a', 1: 'b'},
        label2id={'a': 0, 'b': 1},
        use_hf_pretrain=True
    )
