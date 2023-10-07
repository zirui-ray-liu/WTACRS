

from transformers import BertForSequenceClassification #, RobertaConfig, RobertaTokenizer

def transformer_convert(model, config, tokenizer, model_args, adapter_config=None, lora_config=None, approx_config=None):

    from seq2seq.third_party.models import BertForSequenceClassification as approx_BertForSequenceClassification

    if type(model) == BertForSequenceClassification:

        model = approx_BertForSequenceClassification.from_pretrained(
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
    
    print(model)
    import pdb
    pdb.set_trace()

    return model, config, tokenizer
