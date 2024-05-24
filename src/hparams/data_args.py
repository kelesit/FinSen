import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."},
    )

    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to save the dataset in jsonl format."},
    )

    save_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to save the dataset."},
    )

    cutoff_len: int = field(
        default=1024,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )

    val_size: float = field(
        default=0.1,
        metadata={"help": "The size of the validation set."},
    )

    def __post_init__(self):
        if self.save_path and not os.path.exists(self.save_path):
            ValueError("save_path does not exist")
        if self.jsonl_path and not os.path.exists(self.jsonl_path):
            ValueError("jsonl_path does not exist")

        assert self.val_size > 0 and self.val_size < 1, "val_size should be in (0,1)"