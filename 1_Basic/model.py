import os
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
    logging
)

# Disable W&B and suppress logs
os.environ["WANDB_DISABLED"] = "true"
logging.set_verbosity_error()

# 1. Use small model
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure pad token exists
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# 2. Load training dataset
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

train_dataset = load_dataset("train.txt", tokenizer)

# 3. Training settings
training_args = TrainingArguments(
    output_dir="./distilgpt2-performance",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_total_limit=1,
    prediction_loss_only=True,
    logging_steps=10,
    report_to="none",  # No W&B
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# 4. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# 5. Train and save
trainer.train()
trainer.save_model("./distilgpt2-performance")
tokenizer.save_pretrained("./distilgpt2-performance")

# 6. Inference function
def generate_label(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test it
prompt = "CPU Load: 90%, Memory: 1GB, Disk: 95% -> Label:"
print("Generated:", generate_label(prompt, model, tokenizer))
