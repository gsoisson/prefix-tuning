import datetime
import os.path
from typing import Any, Dict

from inference import generate_text
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from utils import get_logger

from data import prepare_dataset

# Initialize logger
logger = get_logger()


def train_peft_model_w_prefix_tuning(
    train_set_path: str = "../data/train.csv",
    test_set_path: str = "../data/test.csv",
    pretrain_model: str = "openai-community/gpt2",
    model_path: str = "../model/",
    text_col: str = "text",
    batch_size: int = 8,
    n_epochs: int = 2,
    learning_rate: float = 3e-5,
    n_virtual_tokens: int = 20,
):
    """Train a text generation model with prefix tuning in PEFT.

    Args:
        train_set_path:
            CSV path to load raw train data.
        test_set_path:
            CSV path to load raw test data.
        pretrain_model:
            Assign transformers pretrained model path or name
        model_path:
            The output directory where the model predictions and checkpoints will be written
        text_col:
            Text column name
        batch_size:
            Batch size to prepare data and model training
        n_epochs:
            Assign training epochs
        learning_rate:
            Learning rate
        n_virtual_tokens:
            Number of virtual tokens as prefix.

    Returns:
        folder_name:
            Model folder name
    """
    # Prepare transformers dataset from csv(raw data)
    ds = prepare_dataset(train_set_path, test_set_path)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define tokenization function
    def tokenize_text(examples: Dict[str, Any]):
        return tokenizer(
            examples[text_col], truncation=True, max_length=512 - n_virtual_tokens
        )

    # Do preprocessing
    tokenized_posts = ds.map(
        tokenize_text,
        batch_size=batch_size,
        writer_batch_size=batch_size,
        batched=True,
    )

    # Data collator that will dynamically pad the inputs received
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # PromptEncoder
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        num_virtual_tokens=n_virtual_tokens,
    )

    # Create causal language model for text generation
    model = AutoModelForCausalLM.from_pretrained(pretrain_model)

    # Test model before fine-tuning
    logger.info("Test the model before fine-tuning.")
    prompt = "Hello, I'm a language model,"
    generated_text = generate_text(prompt, model, tokenizer)
    logger.info(f"Prompt: {prompt}\nGenerated text: {generated_text[0]}")

    # Get Peft model object from a model and a config
    model = get_peft_model(model, peft_config)

    # Get number of trainable parameters
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )

    # Create output directory
    folder_name = f"{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"
    log_dir = os.path.join(model_path, folder_name)

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=log_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_posts["train"],
        eval_dataset=tokenized_posts["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Execute training
    trainer.train()

    # Save model
    trainer.save_model(log_dir)

    return folder_name
