"""
=============================================================================
 Prompt Tuning PoC -- Streamlit UI
 Paper: "The Power of Scale for Parameter-Efficient Prompt Tuning"
         (Lester, Al-Rfou, Raffel, 2021 -- arXiv:2104.08691)

 Run:  streamlit run streamlit_app.py
=============================================================================
"""

import os
import time
import warnings

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import streamlit as st
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

# -- Configuration ------------------------------------------------------------
MODEL_NAME = "t5-small"
NUM_VIRTUAL_TOKENS = 20
PROMPT_INIT_TEXT = "Classify the sentiment of this sentence as positive or negative:"
NUM_EPOCHS = 10
LEARNING_RATE = 1e-2
BATCH_SIZE = 16
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 8
TRAIN_SAMPLES = 4000
EVAL_SAMPLES = 500
TASK_PREFIX = "sst2 sentence: "
OUTPUT_DIR = "./prompt_tuning_output"
LABEL_MAP = {0: "negative", 1: "positive"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- Page config --------------------------------------------------------------
st.set_page_config(page_title="Prompt Tuning PoC", layout="wide")

st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-card h3 { color: #cdd6f4; margin: 0 0 4px 0; font-size: 14px; }
    .metric-card p  { color: #89b4fa; margin: 0; font-size: 24px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
#  CACHED MODEL + DATASET LOADING
# ==============================================================================
@st.cache_resource(show_spinner="Loading T5-Small model...")
def load_model():
    """Load and return (tokenizer, peft_model)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        prompt_tuning_init_text=PROMPT_INIT_TEXT,
        tokenizer_name_or_path=MODEL_NAME,
    )

    model = get_peft_model(base_model, peft_config).to(DEVICE)
    return tokenizer, model


@st.cache_resource(show_spinner="Loading SST-2 dataset...")
def load_data(_tokenizer):
    """Load and preprocess SST-2. The underscore prefix tells Streamlit not to hash the tokenizer."""
    dataset = load_dataset("stanfordnlp/sst2")
    raw_train = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
    raw_eval = dataset["validation"].shuffle(seed=42).select(
        range(min(EVAL_SAMPLES, len(dataset["validation"])))
    )

    def preprocess(examples):
        inputs = [TASK_PREFIX + s for s in examples["sentence"]]
        targets = [LABEL_MAP[l] for l in examples["label"]]
        model_inputs = _tokenizer(inputs, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True)
        labels = _tokenizer(targets, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True)
        labels["input_ids"] = [
            [(t if t != _tokenizer.pad_token_id else -100) for t in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = raw_train.map(preprocess, batched=True, remove_columns=raw_train.column_names)
    eval_ds = raw_eval.map(preprocess, batched=True, remove_columns=raw_eval.column_names)
    return train_ds, eval_ds, dataset


# ==============================================================================
#  HELPERS
# ==============================================================================
def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def predict(text, tokenizer, model):
    model.eval()
    prefixed = TASK_PREFIX + text
    inputs = tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_TARGET_LENGTH)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()


# ==============================================================================
#  LOAD RESOURCES
# ==============================================================================
tokenizer, model = load_model()
train_ds, eval_ds, raw_dataset = load_data(tokenizer)


# ==============================================================================
#  SIDEBAR -- SYSTEM INFO
# ==============================================================================
with st.sidebar:
    st.header("System Info")
    st.text(f"Device:  {DEVICE}")
    if DEVICE.type == "cuda":
        st.text(f"GPU:     {torch.cuda.get_device_name(0)}")
        st.text(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        st.text("GPU:     Not available")

    st.divider()
    st.header("Config")
    st.text(f"Model:          {MODEL_NAME}")
    st.text(f"Virtual Tokens: {NUM_VIRTUAL_TOKENS}")
    st.text(f"Epochs:         {NUM_EPOCHS}")
    st.text(f"Learning Rate:  {LEARNING_RATE}")
    st.text(f"Train Samples:  {TRAIN_SAMPLES}")
    st.text(f"Batch Size:     {BATCH_SIZE}")


# ==============================================================================
#  MAIN LAYOUT
# ==============================================================================
st.title("Prompt Tuning PoC")
st.caption("Paper: The Power of Scale for Parameter-Efficient Prompt Tuning")

tab_params, tab_train, tab_eval, tab_predict = st.tabs([
    "Parameter Comparison", "Train", "Evaluate", "Predict"
])


# -- TAB 1: Parameter Comparison -----------------------------------------------
with tab_params:
    st.subheader("Full Fine-Tuning vs Prompt Tuning")

    pt_trainable, pt_total = count_params(model)
    full_ft = pt_total
    reduction = full_ft / pt_trainable if pt_trainable > 0 else 0
    pt_pct = pt_trainable / pt_total * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Full FT Params", f"{full_ft:,}", "100%")
    col2.metric("Prompt Tuning Params", f"{pt_trainable:,}", f"{pt_pct:.4f}%")
    col3.metric("Reduction Factor", f"{reduction:,.0f}x", "fewer params")
    col4.metric("Params Saved", f"{100 - pt_pct:.4f}%", "frozen")

    st.divider()
    st.subheader("Engineering Analysis")

    comparison_data = {
        "Feature": [
            "Trainable Params",
            "Trainable %",
            "Training Speed",
            "Memory Usage",
            "Storage per Task",
            "Overfitting Risk",
            "Multi-task Serving",
        ],
        "Full Fine-Tuning": [
            f"{full_ft:,}",
            "100.0000%",
            "Slow (all layers)",
            "High",
            "~242 MB",
            "Higher",
            "N model copies",
        ],
        "Prompt Tuning": [
            f"{pt_trainable:,}",
            f"{pt_pct:.4f}%",
            "Fast (prompt only)",
            "Very Low",
            "~16 KB",
            "Lower",
            "1 model + N prompts",
        ],
    }
    st.table(comparison_data)


# -- TAB 2: Training -----------------------------------------------------------
with tab_train:
    st.subheader("Train Soft Prompt on SST-2")

    st.text(f"Dataset: {TRAIN_SAMPLES} train samples  |  Model: {MODEL_NAME} (frozen)")
    st.text(f"Trainable: {pt_trainable:,} / {pt_total:,} params ({pt_pct:.4f}%)")

    if st.button("Start Training", type="primary"):
        progress_bar = st.progress(0, text="Preparing trainer...")

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

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=data_collator,
        )

        progress_bar.progress(10, text="Training in progress...")
        start = time.time()
        trainer.train()
        elapsed = time.time() - start

        progress_bar.progress(100, text="Training complete!")
        st.session_state["trained"] = True

        st.success(f"Training finished in {elapsed:.1f}s")


# -- TAB 3: Evaluation ---------------------------------------------------------
with tab_eval:
    st.subheader("Evaluate on SST-2 Validation Set")

    if not st.session_state.get("trained", False):
        st.info("Train the model first using the Train tab.")
    else:
        if st.button("Run Evaluation", type="primary"):
            model.eval()
            correct = 0
            total = 0
            batch_size = 16

            raw_eval = raw_dataset["validation"].shuffle(seed=42).select(
                range(min(EVAL_SAMPLES, len(raw_dataset["validation"])))
            )

            progress = st.progress(0, text="Evaluating...")

            for i in range(0, len(raw_eval), batch_size):
                batch = raw_eval[i : i + batch_size]
                prefixed = [TASK_PREFIX + s for s in batch["sentence"]]
                inputs = tokenizer(
                    prefixed, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_INPUT_LENGTH,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=MAX_TARGET_LENGTH)

                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for pred, label in zip(preds, batch["label"]):
                    if pred.strip().lower() == LABEL_MAP[label]:
                        correct += 1
                    total += 1

                progress.progress(
                    min(i + batch_size, len(raw_eval)) / len(raw_eval),
                    text=f"Processed {min(i + batch_size, len(raw_eval))}/{len(raw_eval)}",
                )

            accuracy = correct / total * 100 if total > 0 else 0
            progress.progress(1.0, text="Evaluation complete!")

            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{accuracy:.2f}%")
            col2.metric("Correct", f"{correct} / {total}")


# -- TAB 4: Custom Prediction --------------------------------------------------
with tab_predict:
    st.subheader("Test Custom Text")

    if not st.session_state.get("trained", False):
        st.info("Train the model first using the Train tab.")
    else:
        user_text = st.text_area(
            "Enter a sentence or paragraph to classify:",
            height=120,
            placeholder="e.g. This movie was absolutely wonderful and I loved every minute of it.",
        )

        if st.button("Classify Sentiment", type="primary") and user_text.strip():
            result = predict(user_text.strip(), tokenizer, model)
            if "positive" in result:
                st.success(f"Prediction: {result.upper()}")
            elif "negative" in result:
                st.error(f"Prediction: {result.upper()}")
            else:
                st.warning(f"Prediction: {result}")
