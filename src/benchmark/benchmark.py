
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from tfns import test_tfns


def run_benchmark():
    ## load the chatglm2-6b base model
    base_model = "THUDM/chatglm3-6b"
    peft_model = "../saves/chatglm3-7b/lora/sft"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, load_in_8bit=True, device_map = "cuda:0")

    model = PeftModel.from_pretrained(model, peft_model)

    model = model.eval()

    res = test_tfns(model, tokenizer, batch_size = 8)

if __name__ == '__main__':
    run_benchmark()