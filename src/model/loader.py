
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PretrainedConfig, PreTrainedTokenizer
from peft import prepare_model_for_kbit_training

from .patcher import patch_config
from .adapter import init_adapter
from ..hparams.model_args import ModelArguments
from ..hparams.finetuning_args import FinetuningArguments



def load_tokenizer(model_args) -> "PreTrainedTokenizer":
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    return tokenizer

def load_config(model_args:"ModelArguments") -> "PretrainedConfig":
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    return config

def load_model(
        tokenizer:"AutoTokenizer",
        model_args:"ModelArguments",
        finetuning_args:"FinetuningArguments",
        is_trainable: bool = False,
        ):
    
    init_kwargs = {
        "trust_remote_code": True,
    }
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

    model = AutoModelForCausalLM.from_pretrained(**init_kwargs)
    if is_trainable and model_args.quantization_bit == 4:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()
        model.print_trainable_parameters()
        
    return model
    