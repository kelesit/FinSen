import json
from transformers import AutoTokenizer, AutoConfig

def preprocess_example(tokenizer: "AutoTokenizer", config: "AutoConfig", example, max_seq_length):
    prompt = example['context']
    target = example['target']
    prompt_ids = tokenizer.encode(prompt, max_seq_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(target, max_seq_length=max_seq_length, truncation=True, add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

def preprocess(path, tokenizer:"AutoTokenizer", config: "AutoConfig", max_seq_length, skip_overlength=False):
    with open(path, 'r') as f:
        for line in f.readlines():
            example = json.loads(line)
            feature = preprocess_example(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature