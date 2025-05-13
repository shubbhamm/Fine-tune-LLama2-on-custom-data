import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling
import torch


# 1. Load your dataset
df = pd.read_csv("cnbc_headlines.csv")
df = df.rename(columns={"Headlines": "text"})
df = df.dropna(subset=["text"])
df = df[["text"]]

# 3. Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1)

# 4. Tokenizer
model_name = "NousResearch/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos for llama

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# 5. Load model with LoRA support
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # safer for ROCm
)

# 6. PEFT LoRA Config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned-llama",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    # evaluation_strategy="no",
    save_strategy="epoch",
    logging_steps=20,
    learning_rate=2e-5,
    fp16=True,
    push_to_hub=False
)

# 8. Trainer setup
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 9. Train the model
trainer.train()

# 10. Save the model and tokenizer
trainer.save_model("./finetuned-llama")
tokenizer.save_pretrained("./finetuned-llama")