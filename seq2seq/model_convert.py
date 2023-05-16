

from transformers import RobertaForSequenceClassification, RobertaConfig

def transformer_convert(model, config, model_args, adapter_config=None, lora_config=None, approx_config=None):

    from seq2seq.third_party.models import RobertaConfig as approx_RobertaConfig
    from seq2seq.third_party.models import RobertaForSequenceClassification as approx_RobertaForSequenceClassification

    if type(model) == RobertaForSequenceClassification:

        config_ = approx_RobertaConfig.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        model_ = approx_RobertaForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            adapter_config=adapter_config,
            lora_config=lora_config,
            approx_config=approx_config,
        )

    else:

        raise NotImplementedError

    del model, config
    return model_, config_
