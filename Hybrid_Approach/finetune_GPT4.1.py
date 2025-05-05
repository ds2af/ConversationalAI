# Demo: Fine-tuning Gemma-3-4b-it with a custom dataset from JSON
# (Disaster Tweet Classification Explanations)
# and then saving, reloading, and querying the fine-tuned model.

# ---------------------------
# Step 1: Load Pre-trained Model and Tokenizer
# ---------------------------
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset, load_dataset # Added load_dataset
from trl import SFTTrainer, SFTConfig
import torch
from transformers import TextStreamer
import gc         # For garbage collection
import sys        # For sys.exit on error
import traceback  # For detailed error printing
import json       # For loading the JSON file manually

print("Step 1: Loading Base Model and Tokenizer...")
# Load the pre-trained Gemma-3-4b-it model with 4-bit quantization
# Ensure you have logged into Hugging Face if needed: `huggingface-cli login`
try:
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-4b-it", # Using the Unsloth optimized version
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,      # Set to False if load_in_4bit is True
        full_finetuning=False    # Use PEFT (LoRA)
    )
except Exception as e:
    print(f"\n--- ERROR DURING BASE MODEL LOADING ---")
    print(f"Error message: {e}")
    print("\nCheck your internet connection, Hugging Face login status (`huggingface-cli login`),")
    print("and ensure the model name 'unsloth/gemma-3-4b-it' is correct and accessible.")
    print("Also check VRAM availability.")
    traceback.print_exc()
    sys.exit(1)
print("Base model loaded.")

# Prepare the model for PEFT fine-tuning (using LoRA-like settings)
print("Applying PEFT/LoRA adaptations...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # Focus only on text layers
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,                      # Increased LoRA rank slightly for potentially more complex task
    lora_alpha=16,             # Match LoRA alpha to rank (common practice)
    lora_dropout=0.05,         # Slightly increased dropout for regularization with more data
    bias="none",
    random_state=3407,         # Seed for reproducibility
)
print("PEFT model prepared.")

# Apply the "gemma-3" chat template to the tokenizer
print("Applying chat template to tokenizer...")
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
print("Tokenizer ready.")

# ---------------------------
# Step 2: Load Custom Dataset from JSON
# ---------------------------
print("\nStep 2: Loading Custom Dataset from JSON...")
dataset_path = "DataPrepForRefinement/finetuning_conversations.json" # Your file name

try:
    # Load the JSON file manually first to ensure structure is correct
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Verify the expected structure: {"conversations": [ [conv1], [conv2], ... ]}
    if "conversations" not in raw_data or not isinstance(raw_data["conversations"], list):
         raise ValueError("JSON file must contain a top-level key 'conversations' mapped to a list of conversations.")
    if not all(isinstance(conv, list) for conv in raw_data["conversations"]):
         raise ValueError("Each item under 'conversations' must be a list representing a single conversation.")
    if not all(isinstance(turn, dict) and "role" in turn and "content" in turn
               for conv in raw_data["conversations"] for turn in conv):
         raise ValueError("Each turn within a conversation must be a dictionary with 'role' and 'content' keys.")

    # Create a Dataset object using the `datasets` library.
    # The expected input for from_dict is a dictionary where keys are column names
    # and values are lists of the data for that column.
    # Our data is already in the format {"conversations": [ [conv1_turns], [conv2_turns], ... ]}
    train_dataset = Dataset.from_dict(raw_data) # Pass the loaded dictionary directly

    print(f"Dataset loaded successfully with {len(train_dataset)} conversations.")

except FileNotFoundError:
    print(f"Error: Dataset file not found at '{dataset_path}'")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{dataset_path}'. Check file integrity.")
    sys.exit(1)
except ValueError as ve:
    print(f"Error: Dataset structure validation failed: {ve}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during dataset loading: {e}")
    traceback.print_exc()
    sys.exit(1)


# The chat template converts our list of dictionaries into a single text string
# suitable for the SFTTrainer's 'dataset_text_field'.
# Reusing the function from the original script
def apply_chat_template(examples):
    formatted_texts = []
    for conv in examples["conversations"]:
        try:
            # Ensure input to apply_chat_template is a list of dicts
            if not isinstance(conv, list) or not all(isinstance(turn, dict) for turn in conv):
                 print(f"Skipping invalid conversation format: {conv}") # Debug print
                 formatted_texts.append(None) # Or handle error appropriately
                 continue
            formatted_texts.append(
                tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=False # Important for training data
                )
            )
        except Exception as e:
            print(f"Error formatting conversation: {conv}")
            print(f"Error: {e}")
            formatted_texts.append(None) # Handle error

    # Filter out potential None values if errors occurred
    return {"text": [text for text in formatted_texts if text is not None]}

# Apply the mapping function. Use multiple processes if CPU allows.
# Consider increasing num_proc if mapping is slow
print("Applying chat template to dataset...")
train_dataset = train_dataset.map(
    apply_chat_template,
    batched=True,
    num_proc=4, # Adjust based on your CPU cores
    remove_columns=["conversations"] # Remove original column after formatting
)
print(f"Dataset formatted. Remaining examples after potential errors: {len(train_dataset)}")


# Check if dataset is empty after processing
if len(train_dataset) == 0:
    print("Error: No valid conversations remained after formatting. Check data and formatting function.")
    sys.exit(1)

print("\nSample formatted text for training:")
try:
    print(train_dataset[0]['text'])
except IndexError:
    print("Could not display sample text - dataset might be empty or structured unexpectedly.")
print("Dataset ready for training.")


# ---------------------------
# Step 3: Fine-Tuning with SFTTrainer
# ---------------------------
print("\nStep 3: Setting up SFTTrainer...")

# Calculate approximate steps per epoch
EFFECTIVE_BATCH_SIZE = 8 # Adjust if needed (per_device_train_batch_size * gradient_accumulation_steps)
steps_per_epoch = (len(train_dataset) + EFFECTIVE_BATCH_SIZE - 1) // EFFECTIVE_BATCH_SIZE

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=None, # No evaluation dataset specified
    args=SFTConfig(
        dataset_text_field="text",         # Field containing the formatted text
        per_device_train_batch_size=2,     # Adjust based on VRAM (aim for largest power of 2 that fits)
        gradient_accumulation_steps=4,     # Accumulate gradients (effective batch size = 2 * 4 = 8)
        warmup_steps=max(1, int(0.1 * steps_per_epoch)), # Warmup for ~10% of first epoch steps
        # max_steps = -1,                  # Set num_train_epochs instead for epoch-based training
        num_train_epochs=3,                # Train for 1 full epoch (adjust if needed)
        learning_rate=2e-4,                # Standard LoRA learning rate
        logging_steps=25,                  # Log metrics every 25 steps
        optim="adamw_8bit",                # Use 8-bit AdamW optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",        # Linear learning rate decay
        seed=3407,
        report_to="none",                  # Disable external reporting (like WandB)
        output_dir="outputs",              # Directory for checkpoints and logs
        save_strategy="epoch",             # Save a checkpoint at the end of each epoch
        # save_steps = 100,                # Or save every N steps if preferred over epochs
        save_total_limit=1,                # Keep only the last checkpoint
    ),
)

print(f"\nStarting fine-tuning on {len(train_dataset)} examples for {trainer.args.num_train_epochs} epoch(s)...")
print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}")
print(f"Approximate steps per epoch: {steps_per_epoch}")
print(f"Total approximate steps: {steps_per_epoch * trainer.args.num_train_epochs}")
print(f"Warmup steps: {trainer.args.warmup_steps}")

try:
    trainer_stats = trainer.train()
    print("Fine-tuning complete. Stats:", trainer_stats)
except Exception as e:
    print(f"\n--- ERROR DURING TRAINING ---")
    print(f"Error message: {e}")
    print("\nCheck VRAM usage (nvidia-smi), batch size configuration, and dataset integrity.")
    traceback.print_exc()
    sys.exit(1)


# ---------------------------
# Step 4: Save the Fine-Tuned Model (LoRA Adapters)
# ---------------------------
print("\nStep 4: Saving Fine-tuned Model Adapters...")
custom_model_dir = "gemma3_train_with_GPT4.1" # <--- Your desired save directory name
try:
    model.save_pretrained(custom_model_dir)
    tokenizer.save_pretrained(custom_model_dir)
    print(f"Model adapters and tokenizer saved to directory: {custom_model_dir}")
except Exception as e:
    print(f"\n--- ERROR DURING SAVING ---")
    print(f"Error message: {e}")
    traceback.print_exc()
    # Continue to cleanup even if saving fails

# ---------> MEMORY CLEANUP <---------
print("\nAttempting to clean up memory before reloading...")
# Delete variables holding the model and trainer objects
del model
del tokenizer
del trainer
del train_dataset # Also remove dataset object
# Force Python garbage collection
gc.collect()
# Empty PyTorch's CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
# Run garbage collection again
gc.collect()
print("Memory cleanup attempt complete.")
# ---------> END MEMORY CLEANUP <---------


# ---------------------------
# Step 5: Reload the Fine-Tuned Model
# ---------------------------
print("\nStep 5: Reloading the base model with fine-tuned adapters...")
try:
    # Check if save directory exists before loading
    import os
    if not os.path.isdir(custom_model_dir):
        print(f"Error: Saved model directory '{custom_model_dir}' not found. Cannot reload.")
        sys.exit(1)

    fine_tuned_model, fine_tuned_tokenizer = FastModel.from_pretrained(
        custom_model_dir, # Load from the directory containing adapters
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        device_map="auto", # Keep device_map auto
    )
    # Reapply the chat template to the reloaded tokenizer (good practice)
    fine_tuned_tokenizer = get_chat_template(fine_tuned_tokenizer, chat_template="gemma-3")
    print("Model reloaded successfully.")

except Exception as e:
    print(f"\n--- ERROR DURING MODEL RELOAD ---")
    print(f"Error message: {e}")
    print("\nTraceback:")
    traceback.print_exc()
    print("\nCheck VRAM usage with nvidia-smi. Ensure the adapters were saved correctly.")
    print(f"Verify the contents of the '{custom_model_dir}' directory.")
    print("\nExiting due to reload error.")
    sys.exit(1) # Exit the script if reload fails

# ---------------------------
# Step 6: Test the Fine-Tuned Model
# ---------------------------
print("\nStep 6: Testing the Fine-tuned Model...")
# Test with a question relevant to the fine-tuning task (Disaster Tweet Classification Explanation)

# Example 1: Potentially disaster-related
messages1 = [{
    "role": "user",
    "content": "Why might the following tweet be considered disaster-related? Tweet: \"BREAKING: Massive earthquake reported near the coast. Tsunami warning issued.\""
}]

# Example 2: Potentially non-disaster-related
messages2 = [{
    "role": "user",
    "content": "Why is the following tweet likely not disaster-related? Tweet: \"Ugh, my fantasy football team is a disaster this week! #fantasyfail\""
}]

# Choose which message set to test
test_messages = messages1 # Or change to messages2

# Convert the test conversation prompt using the chat template.
# Add the generation prompt to signal the model to start responding.
print(f"Formatting test prompt: {' '.join([turn['content'] for turn in test_messages])[:100]}...")
try:
    prompt_text = fine_tuned_tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,            # Output is a string
        add_generation_prompt=True # Crucial for inference
    )
except Exception as e:
    print(f"\n--- ERROR FORMATTING TEST PROMPT ---")
    print(f"Error message: {e}")
    print(f"Problematic messages structure: {test_messages}")
    traceback.print_exc()
    sys.exit(1)


print("\nFormatted prompt for generation:")
print(prompt_text)

# Tokenize the prompt for the model
print("Tokenizing prompt...")
inputs = fine_tuned_tokenizer([prompt_text], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

# Generate output: use a TextStreamer for interactive display.
print("\nGenerating response from the fine-tuned model...")
try:
    outputs = fine_tuned_model.generate(
        **inputs,
        max_new_tokens=150,          # Increased max tokens slightly for potentially longer explanations
        temperature=0.6,             # Slightly lower temperature for more focused explanation
        top_p=0.9,
        top_k=50,
        do_sample=True,
        streamer=TextStreamer(fine_tuned_tokenizer, skip_prompt=True), # Stream output
    )
except Exception as e:
    print(f"\n--- ERROR DURING GENERATION ---")
    print(f"Error message: {e}")
    traceback.print_exc()


print("\n\n--- Experiment Complete ---")