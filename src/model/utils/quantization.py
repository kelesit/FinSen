from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from ...hparams.model_args import ModelArguments

def configure_quantization(config:"AutoConfig", model_args:'ModelArguments', init_kwargs: Dict[str, Any]):
    if model_args.quantization_bit == 8:
            init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif model_args.quantization_bit == 4:
        init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=model_args.quantization_type,
            bnb_4bit_use_double_quant=model_args.double_quantization,
        )
