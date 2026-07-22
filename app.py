"""
=============================================================================
 Prompt Tuning PoC — Single File Implementation
 Paper: "The Power of Scale for Parameter-Efficient Prompt Tuning"
         (Lester, Al-Rfou, Raffel, 2021 — arXiv:2104.08691)

 Model:   T5-Small (frozen)
 Dataset: SST-2 (Stanford Sentiment Treebank — binary sentiment)
 Method:  Soft Prompt Tuning via HuggingFace PEFT
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import sys
import time
import warnings

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from datasets import load_dataset

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = "t5-small"
NUM_VIRTUAL_TOKENS = 20         # Soft prompt length (paper recommends 20-100)
PROMPT_INIT_TEXT = "Classify the sentiment of this sentence as positive or negative:"
NUM_EPOCHS = 10
LEARNING_RATE = 1e-2
BATCH_SIZE = 16
MAX_INPUT_LENGTH = 256          # Longer to handle paragraphs
MAX_TARGET_LENGTH = 8
TRAIN_SAMPLES = 4000            # More data for better generalisation
EVAL_SAMPLES = 500              # Subset size for evaluation
TASK_PREFIX = "sst2 sentence: " # T5-style task prefix for better alignment
OUTPUT_DIR = "./prompt_tuning_output"
LABEL_MAP = {0: "negative", 1: "positive"}

# ── Device Detection ─────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global State ─────────────────────────────────────────────────────────────
tokenizer = None
base_model = None
peft_model = None
train_dataset = None
eval_dataset = None
is_trained = False


# =============================================================================
#  1. MODEL SETUP
# =============================================================================
def load_model_and_tokenizer():
    """Load T5-Small and wrap it with a Prompt Tuning configuration."""
    global tokenizer, base_model, peft_model

    print(f"\nLoading T5-Small model and tokenizer...")
    print(f"   Device: {DEVICE} {'(GPU accelerated)' if DEVICE.type == 'cuda' else '(CPU mode)'}")
    if DEVICE.type == 'cuda':
        print(f"   GPU:    {torch.cuda.get_device_name(0)}")
        print(f"   VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Configure Prompt Tuning via PEFT
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,     # Vocab-based init (paper rec.)
        num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        prompt_tuning_init_text=PROMPT_INIT_TEXT,
        tokenizer_name_or_path=MODEL_NAME,
    )

    peft_model = get_peft_model(base_model, peft_config)

    # Move model to GPU if available
    peft_model = peft_model.to(DEVICE)
    print(f"Model loaded on {DEVICE} and Prompt Tuning configured.\n")


# =============================================================================
#  2. DATASET LOADING
# =============================================================================
def load_and_prepare_dataset():
    """Load SST-2 from HuggingFace and prepare train/eval splits."""
    global train_dataset, eval_dataset

    print("Loading SST-2 dataset...")
    dataset = load_dataset("stanfordnlp/sst2")

    # Use subsets for fast training on a laptop
    raw_train = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
    raw_eval = dataset["validation"].shuffle(seed=42).select(
        range(min(EVAL_SAMPLES, len(dataset["validation"])))
    )

    def preprocess(examples):
        """Convert SST-2 examples into T5 text-to-text format."""
        # T5 expects text input with task prefix and text output
        inputs = [TASK_PREFIX + sent for sent in examples["sentence"]]
        targets = [LABEL_MAP[label] for label in examples["label"]]

        model_inputs = tokenizer(
            inputs, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True
        )
        labels = tokenizer(
            targets, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True
        )
        # Replace padding token id with -100 so it's ignored in loss
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = raw_train.map(preprocess, batched=True, remove_columns=raw_train.column_names)
    eval_dataset = raw_eval.map(preprocess, batched=True, remove_columns=raw_eval.column_names)

    print(f"✅ Dataset ready — {len(train_dataset)} train / {len(eval_dataset)} eval samples.\n")


# =============================================================================
#  3. PARAMETER COUNTING
# =============================================================================
def count_parameters(model):
    """Return (trainable_params, total_params) for a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def display_parameter_comparison():
    """Print a side-by-side comparison of Full FT vs Prompt Tuning params."""
    if peft_model is None:
        print("⚠️  Model not loaded. Run option 1 first.")
        return

    pt_trainable, pt_total = count_parameters(peft_model)

    # For full fine-tuning, all params would be trainable
    full_ft_trainable = pt_total

    reduction_factor = full_ft_trainable / pt_trainable if pt_trainable > 0 else 0
    pt_percent = (pt_trainable / pt_total) * 100

    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  PARAMETER COMPARISON: Full Fine-Tuning vs Prompt Tuning ".ljust(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print("║" + f"  Full Fine-Tuning Trainable:  {full_ft_trainable:>14,}  (100.00%)".ljust(58) + "║")
    print("║" + f"  Prompt Tuning Trainable:     {pt_trainable:>14,}  ({pt_percent:.4f}%)".ljust(58) + "║")
    print("║" + "─" * 56 + "  ║")
    print("║" + f"  Parameter Reduction:         {reduction_factor:>14,.0f}×  fewer".ljust(58) + "║")
    print("║" + f"  Parameters Saved:            {(100 - pt_percent):>13.4f}%".ljust(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print("║" + "                                                          " + "║")

    # Engineering Analysis Table
    rows = [
        ("Feature",            "Full Fine-Tuning",     "Prompt Tuning"),
        ("─" * 20,             "─" * 20,               "─" * 14),
        ("Trainable Params",   f"{full_ft_trainable:,}", f"{pt_trainable:,}"),
        ("Trainable %",        "100.0000%",            f"{pt_percent:.4f}%"),
        ("Training Speed",     "Slow (all layers)",    "Fast (prompt only)"),
        ("Memory Usage",       "High",                 "Very Low"),
        ("Storage per Task",   "~242 MB",              "~16 KB"),
        ("Overfitting Risk",   "Higher",               "Lower"),
        ("Multi-task Serving",  "N model copies",      "1 model + N prompts"),
    ]

    for feat, full, pt in rows:
        line = f"  {feat:<20} {full:<22} {pt}"
        print("║" + line.ljust(58) + "║")

    print("║" + "                                                          " + "║")
    print("╚" + "═" * 58 + "╝")
    print()


# =============================================================================
#  4. TRAINING
# =============================================================================
def train_model():
    """Train the soft prompt on SST-2."""
    global is_trained

    if peft_model is None or train_dataset is None:
        print("⚠️  Model or dataset not loaded. Please wait for initialisation.")
        return

    print("\nStarting Prompt Tuning Training...")
    print(f"   Epochs: {NUM_EPOCHS}  |  LR: {LEARNING_RATE}  |  Batch: {BATCH_SIZE}")
    print(f"   Virtual Tokens: {NUM_VIRTUAL_TOKENS}  |  Init: \"{PROMPT_INIT_TEXT}\"")

    pt_trainable, pt_total = count_parameters(peft_model)
    print(f"   Trainable params: {pt_trainable:,} / {pt_total:,} ({pt_trainable/pt_total*100:.4f}%)\n")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=peft_model)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    is_trained = True
    print(f"\n✅ Training complete in {elapsed:.1f}s.")
    print(f"   The frozen T5 model is now conditioned for sentiment analysis.\n")


# =============================================================================
#  5. EVALUATION
# =============================================================================
def evaluate_model():
    """Evaluate accuracy on the SST-2 validation set."""
    if peft_model is None or eval_dataset is None:
        print("⚠️  Model or dataset not loaded.")
        return

    if not is_trained:
        print("⚠️  Model not yet trained. Run option 1 first.")
        return

    print("\n Evaluating on SST-2 validation set...")

    peft_model.eval()
    correct = 0
    total = 0

    # Process evaluation in small batches
    batch_size = 16
    raw_eval = load_dataset("stanfordnlp/sst2", split="validation").shuffle(seed=42).select(
        range(min(EVAL_SAMPLES, 872))
    )

    for i in range(0, len(raw_eval), batch_size):
        batch = raw_eval[i : i + batch_size]
        sentences = batch["sentence"]
        labels = batch["label"]

        # Add task prefix — same as training
        prefixed = [TASK_PREFIX + s for s in sentences]

        inputs = tokenizer(
            prefixed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        inputs = {k: v.to(peft_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = peft_model.generate(**inputs, max_new_tokens=MAX_TARGET_LENGTH)

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for pred, true_label in zip(predictions, labels):
            pred_clean = pred.strip().lower()
            expected = LABEL_MAP[true_label]
            if pred_clean == expected:
                correct += 1
            total += 1

        # Progress indicator
        if (i // batch_size) % 5 == 0:
            print(f"   Processed {min(i + batch_size, len(raw_eval))}/{len(raw_eval)} samples...")

    accuracy = correct / total * 100 if total > 0 else 0

    print()
    print("╔" + "═" * 44 + "╗")
    print("║" + "  EVALUATION RESULTS".ljust(44) + "║")
    print("╠" + "═" * 44 + "╣")
    print("║" + f"  Correct:    {correct} / {total}".ljust(44) + "║")
    print("║" + f"  Accuracy:   {accuracy:.2f}%".ljust(44) + "║")
    print("║" + f"  Dataset:    SST-2 Validation".ljust(44) + "║")
    print("║" + f"  Method:     Prompt Tuning (T5-Small)".ljust(44) + "║")
    print("╚" + "═" * 44 + "╝")
    print()

    return accuracy


# =============================================================================
#  6. CUSTOM TEXT PREDICTION
# =============================================================================
def predict_custom_text():
    """Let the user type a sentence and get a sentiment prediction."""
    if peft_model is None:
        print("⚠️  Model not loaded.")
        return

    if not is_trained:
        print("⚠️  Model not trained yet. Run option 1 first.")
        return

    print("\n🔮 Custom Text Prediction")
    print("   Type a sentence and the prompt-tuned model will classify its sentiment.")
    print("   Type 'back' to return to the menu.\n")

    peft_model.eval()

    while True:
        text = input("   Enter text: ").strip()
        if text.lower() in ("back", "exit", "quit", "q"):
            break
        if not text:
            continue

        # Add the same task prefix used during training
        prefixed_text = TASK_PREFIX + text
        inputs = tokenizer(
            prefixed_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH
        )
        inputs = {k: v.to(peft_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = peft_model.generate(**inputs, max_new_tokens=MAX_TARGET_LENGTH)

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

        # Display with emoji
        if "positive" in prediction:
            emoji = "😊"
        elif "negative" in prediction:
            emoji = "😞"
        else:
            emoji = "🤔"

        print(f"   → Prediction: {prediction} {emoji}\n")


# =============================================================================
#  7. INTERACTIVE CLI MENU
# =============================================================================
def print_banner():
    """Display the application banner."""
    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + "                                                              " + "║")
    print("║" + "   🧠  PROMPT TUNING PoC                                      " + "║")
    print("║" + "   Paper: The Power of Scale for Parameter-Efficient           " + "║")
    print("║" + "          Prompt Tuning (Lester et al., 2021)                  " + "║")
    print("║" + "                                                              " + "║")
    print("║" + "   Model: T5-Small (frozen) + Soft Prompt                     " + "║")
    print("║" + "   Task:  Sentiment Analysis on SST-2                         " + "║")
    print("║" + "                                                              " + "║")
    print("╚" + "═" * 62 + "╝")
    print()


def print_menu():
    """Display the interactive menu."""
    print("╔" + "═" * 50 + "╗")
    print("║" + "     PROMPT TUNING PoC — Interactive Menu          " + "║")
    print("╠" + "═" * 50 + "╣")
    print("║" + "                                                  " + "║")
    print("║" + "  1. Train Prompt-Tuned Model                  " + "║")
    print("║" + "  2. Evaluate Model on Test Set                " + "║")
    print("║" + "  3. Compare Parameters (Full FT vs PT)        " + "║")
    print("║" + "  4. Test Custom Text                          " + "║")
    print("║" + "  5. Exit                                      " + "║")
    print("║" + "                                               " + "║")
    print("╚" + "═" * 50 + "╝")


def main():
    """Main entry point — initialise and run the interactive menu."""
    print_banner()

    # Initialise model and dataset on startup
    load_model_and_tokenizer()
    load_and_prepare_dataset()

    # Show initial parameter info
    display_parameter_comparison()

    while True:
        print_menu()
        choice = input("\n  Select option (1-5): ").strip()

        if choice == "1":
            train_model()
        elif choice == "2":
            evaluate_model()
        elif choice == "3":
            display_parameter_comparison()
        elif choice == "4":
            predict_custom_text()
        elif choice == "5":
            print("\n👋 Goodbye! Keep exploring Parameter-Efficient Fine-Tuning.\n")
            sys.exit(0)
        else:
            print("⚠️  Invalid option. Please enter 1-5.\n")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
