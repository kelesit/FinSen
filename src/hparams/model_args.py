from typing import Optional, Literal
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='THUDM/chatglm3-6b',
        metadata={
            "help": "The pretrained model name or path"
        }   ,
    )

    quantization_bit: Optional[int] = field(
        default=None,
        metadata={
            "help": "The quantization bit"
        },
    )

    double_quantization: bool = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in int4 training."},
    )

    quantization_type: Literal["nf4", "fp4"] = field(
        default='nf4',
        metadata={
            "help": "The quantization type to use in int4 training."
        },
    )

    export_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The directory to save the exported model."
        },
    )

    # def __post_init__(self):
    #     assert self.quantization_bit in [None, 8, 4]

