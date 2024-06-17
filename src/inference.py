import torch
from peft import PeftConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from utils import get_logger

# Initialize logger
logger = get_logger()


def load_model_and_tokenizer(model_dir):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the PEFT configuration
    config = PeftConfig.from_pretrained(model_dir)

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

    # Apply PEFT configuration to the model
    model = get_peft_model(model, config)

    # Load the adapter weights
    model.load_adapter(model_dir, adapter_name="default", config=config)

    return tokenizer, model


def generate_text(prompt, model, tokenizer, max_length=50):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and print the generated texts
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]

    return generated_texts


def run_inference(prompt, model_dir: str):
    # Log the start of the inference process
    logger.info(f"Running inference with model directory: {model_dir}")

    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_dir)

    # Generate text
    generated_text = generate_text(prompt, model, tokenizer)

    logger.info(f"Prompt: {prompt}\nGenerated text: {generated_text[0]}")
