import os
from datetime import datetime, timedelta
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
    logging
)

# Disable W&B and suppress extra logs
os.environ["WANDB_DISABLED"] = "true"
logging.set_verbosity_error()

# Load tokenizer and model
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure pad token exists
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Check for recent HighLoad issues
def check_recent_issues(file_path):
    issues_by_cluster = {}
    issues_by_computer = {}
    now = datetime.now()
    ten_minutes_ago = now - timedelta(minutes=10)

    with open(file_path, "r") as f:
        for line in f:
            try:
                ts_str = line.split("]")[0][1:]
                timestamp = datetime.fromisoformat(ts_str)
                if timestamp < ten_minutes_ago:
                    continue

                if "HighLoad" in line:
                    cluster = line.split("Cluster: ")[1].split(",")[0].strip()
                    computer = line.split("Computer: ")[1].split(",")[0].strip()

                    issues_by_cluster[cluster] = issues_by_cluster.get(cluster, 0) + 1
                    issues_by_computer[computer] = issues_by_computer.get(computer, 0) + 1
            except Exception as e:
                continue  # Skip malformed lines

    if issues_by_cluster or issues_by_computer:
        print("⚠️  Recent HighLoad activity (past 10 minutes):")
        if issues_by_cluster:
            print("Clusters:")
            for cluster, count in issues_by_cluster.items():
                print(f" - {cluster}: {count} alerts")
        if issues_by_computer:
            print("Computers:")
            for comp, count in issues_by_computer.items():
                print(f" - {comp}: {count} alerts")
    else:
        print("✅ No recent HighLoad issues in the past 10 minutes.")

# Call the check before training
check_recent_issues("traincluster.txt")

# Load dataset
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

train_dataset = load_dataset("traincluster.txt", tokenizer)

# Training settings
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

# Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# Train and save
trainer.train()
trainer.save_model("./distilgpt2-performance")
tokenizer.save_pretrained("./distilgpt2-performance")

# Inference example
def generate_label(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the trained model
prompt = "CPU Load: 92%, Memory: 2GB, Disk: 94% -> Label:"
print("Generated:", generate_label(prompt, model, tokenizer))
