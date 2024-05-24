import os
import shutil
import json
from tqdm import tqdm

from datasets import load_dataset
import datasets
from transformers import AutoTokenizer

from ..hparams.model_args import ModelArguments
from ..hparams.data_args import DataArguments


def get_dataset(model_args:"ModelArguments", data_args:"DataArguments", stage, tokenizer:"AutoTokenizer", skip_overlength=False):
    if stage == "sft":
        if data_args.save_path is None:
            if data_args.jsonl_path is None:
                pass
        else:
            dataset = datasets.load_from_disk(data_args.save_path)
            dataset = dataset.train_test_split(test_size=data_args.val_size, shuffle=True, seed = 42)

    else:
        raise ValueError("Only SFT is supported for now")
    
    return dataset
