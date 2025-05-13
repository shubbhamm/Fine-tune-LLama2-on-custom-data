# 🦙 Fine-Tuning LLaMA 2 (7B) on CNBC Headlines using LoRA

This project demonstrates how to fine-tune the [NousResearch/Llama-2-7b-hf](https://huggingface.co/NousResearch/Llama-2-7b-hf) model on a dataset of CNBC news headlines using LoRA (Low-Rank Adaptation) with Hugging Face Transformers and PEFT.

---

## 📁 Dataset

- **File**: `cnbc_headlines.csv`
- **Column used**: `Headlines` (renamed internally to `text`)
- The dataset is split into **train (90%)** and **test (10%)**.

---

## 🔧 Installation

### 💻 Environment Requirements

- Python 3.8+
- Linux (recommended for ROCm/AMD GPUs)
- PyTorch with ROCm (for AMD GPUs) or CUDA (for NVIDIA GPUs)

### 📦 Install Required Libraries

```bash
pip install -U pandas datasets torch torchvision torchaudio \
transformers peft bitsandbytes
```

If you're using AMD GPUs with ROCm 6.1:

```bash
pip install --upgrade torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1/
```

---

## 🚀 Training Script Summary

1. **Load and Clean Data**
2. **Tokenize Text**
3. **Load Pretrained LLaMA 2 Model**
4. **Configure LoRA (PEFT)**
5. **Train using Hugging Face `Trainer` API**
6. **Save the fine-tuned model**

---

## 🔁 LoRA Config Used

```python
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

---

## 🧠 Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./finetuned-llama",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    save_strategy="epoch",
    logging_steps=20,
    learning_rate=2e-5,
    fp16=True,
    push_to_hub=False
)
```

---

## 💾 Output

After training, the fine-tuned model and tokenizer will be saved in:

```
./finetuned-llama/
```

---

## 📌 Notes

- If you face `RuntimeError: No HIP GPUs are available`, your ROCm or AMD GPU drivers may not be installed correctly.
- For CPU-only execution, load model like:
  ```python
  model.to("cpu")
  ```

---

## 🛠️ To-Do

- [ ] Add evaluation metrics like perplexity or accuracy
- [ ] Create an inference script for generating predictions
- [ ] Optionally push model to Hugging Face Hub

---

## 🙏 Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT Library](https://github.com/huggingface/peft)
- [NousResearch/Llama-2-7b-hf](https://huggingface.co/NousResearch/Llama-2-7b-hf)

---

## 📬 Contact

For questions or collaboration, feel free to open an issue or reach out.
