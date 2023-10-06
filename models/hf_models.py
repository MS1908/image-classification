from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification


def hf_models_factory(
    model_name_or_path,
    labels,
    id2label,
    label2id,
    trust_remote_code=False,
    use_hf_pretrain=False
):
    if use_hf_pretrain:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
        )
