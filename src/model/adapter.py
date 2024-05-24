from transformers import PretrainedConfig, PreTrainedModel
from ..hparams.model_args import ModelArguments
from ..hparams.finetuning_args import FinetuningArguments
from peft import TaskType, LoraConfig, get_peft_model

def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
):
    if is_trainable:
        target_modules = finetuning_args.lora_target
        peft_kwargs = {
                "r": finetuning_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": finetuning_args.lora_alpha,
                "lora_dropout": finetuning_args.lora_dropout,
            }
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            bias='none',
            **peft_kwargs
        )
        model = get_peft_model(model, lora_config)

    return model