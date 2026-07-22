import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from datasets import Dataset
import os

# Suppress wandb or other integrations for a clean terminal output
os.environ["WANDB_DISABLED"] = "true"

print("--- Phase 1: Model Architecture ---")
# 1. Define the base model as discussed in the paper's ablations
model_name = "t5-small" 

print(f"Loading base model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Configure the Soft Prompt (Prompt Tuning)
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8, # The length of the soft prompt
    prompt_tuning_init_text="Classify this text:", # Initializing with vocabulary embeddings
    tokenizer_name_or_path=model_name,
)

# 3. Create the PEFT Model
model = get_peft_model(base_model, peft_config)

# 4. Verify Parameter Efficiency
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}%"
    )

print("\nModel Architecture Ready.")
print_trainable_parameters(model)


print("\n--- Phase 2: Training the Soft Prompt ---")
# 1. Create a minimal labeled dataset (for a downstream task: Sentiment Analysis)
data = {
    "text": ["This product is amazing!", "I completely hate this.", "Best experience ever.", "Terrible, do not buy."],
    "label": ["positive", "negative", "positive", "negative"]
}
dataset = Dataset.from_dict(data)

# 2. Preprocess the data for the T5 text-to-text format
def preprocess_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)
    targets = tokenizer(examples["label"], padding="max_length", truncation=True, max_length=8)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# THE CRUCIAL FIX: Remove the string columns so the data collator doesn't crash
tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])

# 3. Configure Training Arguments
training_args = TrainingArguments(
    output_dir="./prompt_tuning_output",
    learning_rate=0.01,  
    num_train_epochs=10, 
    per_device_train_batch_size=2,
    logging_steps=2,
    save_strategy="no" 
)

# 4. Initialize the Trainer
trainer = Trainer(
    model=model, 
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 5. Execute Training (Backpropagation)
print("Training started... Watch the soft prompt parameters update while T5 remains frozen!")
trainer.train()

print("\nTraining complete! The frozen model is now conditioned for sentiment analysis.")


print("\n--- Phase 3: Inference and Testing ---")
# Put the model in evaluation mode
model.eval()

# Define new, unseen test sentences to prove the model learned the task
test_texts = [
    "I really love this!", 
    "This is the worst thing I've ever bought.",
    "Absolutely fantastic quality.",
    "Do not waste your money."
]

print("Testing the frozen model with the trained soft prompt on new data:\n")

for text in test_texts:
    # 1. Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the same device as the model (e.g., GPU or CPU)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 2. Generate the prediction
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    
    # 3. Decode the tokenized output back into human-readable text
    prediction = tokenizer.decode(outputs, skip_special_tokens=True)
    print(f"Input: '{text}'")
    print(f"Prediction: '{prediction}'\n")