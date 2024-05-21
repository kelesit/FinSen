import os

from loguru import logger
import torch
import datasets
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from data import tfns_train_data, save_dataset, preprocess
from tools import print_trainable_parameters
from trainer import ModifiedTrainer
from data.data_collator import ModifiedDataCollator

from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback


jsonl_path = 'data/tfns_dataset.jsonl'
save_path = 'data/tfns_dataset'

MODEL_NAME = 'THUDM/chatglm3-6b'
MAX_SEQ_LENGTH = 512
SKIP_OVERLENGTH = True
SAVED_MODEL_PATH = 'finetuned_model'
RESUME_FROM_CHECKPOINT = False

def main():

    # Load the tokenizer and model config
    model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )


    # # Prepare the dataset
    # train_data = tfns_train_data()
    # save_dataset(train_data, jsonl_path)


    # # Preprocess the dataset
    # dataset = datasets.Dataset.from_generator(
    #     lambda: preprocess(jsonl_path, tokenizer, config, MAX_SEQ_LENGTH, SKIP_OVERLENGTH),
    # )
    # dataset.save_to_disk(save_path)


    # Training arguments
    training_args = TrainingArguments(
        output_dir=SAVED_MODEL_PATH,
        logging_steps=500,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        save_steps=500,
        fp16=True,
        torch_compile=False,
        load_best_model_at_end=True,
        evaluation_strategy='steps',
        remove_unused_columns=False,
    )


    # Load the model
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=False,
        )
    )

    model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    # LoRA config & Setup
    target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias='none',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    
    resume_from_checkpoint=None
    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, 'adapter_model.bin'
            )
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            logger.info('Restarting from {checkpoint_name}')
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.info('No checkpoint found at {checkpoint_name}')


    # Loading Data
    dataset = datasets.load_from_disk(save_path)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed = 42)

    data_collator = ModifiedDataCollator(
        tokenizer=tokenizer
    )

    ## Training    
    writer = SummaryWriter()
    trainer = ModifiedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,             # Trainer args
        train_dataset=dataset["train"], # Training set
        eval_dataset=dataset["test"],   # Testing set
        data_collator=data_collator,    # Data Collator
        callbacks=[TensorBoardCallback(writer)],
    )
    trainer.train()
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)



    


if __name__ == '__main__':
    main()