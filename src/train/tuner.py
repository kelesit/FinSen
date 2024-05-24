from typing import Any, Dict, Optional

from ..hparams import get_train_args
from .sft import run_sft

def run_exp(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, training_args, finetuning_args= get_train_args(args)

    if finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args)
    else:
        raise ValueError("Only SFT is supported for now")


