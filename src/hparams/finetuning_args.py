from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class LoraArguments:
    r"""
    Arguments for LoRA training
    """
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )

    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."}
    )

    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout rate for LoRA fine-tuning."}
    )

    lora_target: str = field(
        default="all",
        metadata={
            "help": """Name(s) of target modules to apply LoRA.\
            Use commas to separate multiple modules. \
            Use "all" to specify all the linear modules. \
            LLaMA choices: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], \
            BLOOM & Falcon & ChatGLM choices: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], \
            Baichuan choices: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"], \
            Qwen choices: ["c_attn", "attn.c_proj", "w1", "w2", "mlp.c_proj"], \
            InternLM2 choices: ["wqkv", "wo", "w1", "w2", "w3"], \
            Others choices: the same as LLaMA."""

        }
    )

@dataclass
class FinetuningArguments(LoraArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with."""

    stage: Literal["sft","dpo", "orpo"] = field(
        default="sft",
        metadata={
            "help": "The stage of fine-tuning. Choose from sft, dpo, or orpo."
        }
    )