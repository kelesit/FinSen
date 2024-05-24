import os, sys
from typing import Any, Dict, Optional, Tuple

from transformers import HfArgumentParser, TrainingArguments

from .finetuning_args import FinetuningArguments
from .model_args import ModelArguments
from .data_args import DataArguments


_TRAIN_ARGS = [ModelArguments, DataArguments, TrainingArguments, FinetuningArguments]


def _parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return (*parsed_args,)

def _parse_train_args(args: Optional[Dict[str, Any]] = None):
    parser = HfArgumentParser(_TRAIN_ARGS)
    return _parse_args(parser, args)




def get_train_args(args: Optional[Dict[str, Any]] = None):
    model_args, data_args, training_args, finetuning_args = _parse_train_args(args)

    return model_args, data_args, training_args, finetuning_args