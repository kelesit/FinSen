from typing import Optional, Any
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class ModifiedDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None

    def __call__(self, features: list) -> Any:
        len_ids = [len(feature["input_ids"]) for feature in features]
        longest = max(len_ids)
        input_ids = []
        labels_list = []
        for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
            ids = feature["input_ids"]
            seq_len = feature["seq_len"]
            labels = (
                [self.tokenizer.pad_token_id] * (seq_len - 1) + ids[(seq_len - 1) :] + [self.tokenizer.pad_token_id] * (longest - ids_l)
            )
            ids = ids + [self.tokenizer.pad_token_id] * (longest - ids_l)
            _ids = torch.LongTensor(ids)
            labels_list.append(torch.LongTensor(labels))
            input_ids.append(_ids)
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }