import datasets

import torch
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer


from .trainer import SFTTrainer
from ...model.loader import load_tokenizer, load_model
from ...data.loader import get_dataset
from ...data.data_collator import ModifiedDataCollator
from ...hparams.model_args import ModelArguments
from ...hparams.data_args import DataArguments
from ...hparams.finetuning_args import FinetuningArguments


from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback


def run_sft(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "TrainingArguments",
        finetuning_args: "FinetuningArguments",
):
    
    tokenizer = load_tokenizer(model_args)
    dataset = get_dataset(model_args, data_args, stage="sft", tokenizer=tokenizer)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = ModifiedDataCollator(
        tokenizer=tokenizer
    )

    # Initialize Trainer
    writer = SummaryWriter()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        callbacks=[TensorBoardCallback(writer)],
    )

    # Training
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.save_state()
    writer.close()

