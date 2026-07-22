import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit

# 1. Define the base model as discussed in the paper's ablations
model_name = "t5-small" 

print(f"Loading base model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Configure the Soft Prompt (Prompt Tuning)
# This implements the core technique: using backpropagation to tune a small 
# set of parameters rather than discrete text prompts.
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8, # The length of the soft prompt
    prompt_tuning_init_text="Classify this text:", # Initializing with vocabulary embeddings
    tokenizer_name_or_path=model_name,
)

# 3. Create the PEFT Model
# This automatically freezes the massive underlying T5 model and ONLY makes 
# the soft prompt parameters trainable.
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

# Next Steps for your PoC: 
# You would pass this 'model' into a standard Hugging Face Trainer 
# along with a downstream task dataset (like sentiment analysis).