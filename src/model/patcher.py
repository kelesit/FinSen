from typing import Any, Dict
from transformers import AutoConfig, AutoTokenizer

from ..hparams.model_args import ModelArguments
from .utils.quantization import configure_quantization

def patch_config(
        config:"AutoConfig",
        tokenizer:"AutoTokenizer",
        model_args:"ModelArguments",
        init_kwargs: Dict[str, Any],
        is_trainable:bool,
):
    configure_quantization(config, model_args, init_kwargs)