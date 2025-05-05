import json
import os
import time
from openai import OpenAI, RateLimitError, APIError

# --- Configuration ---
INPUT_JSON_FILENAME = "labeled_tweets_run0.json" # <--- Your input JSON file name
OUTPUT_JSON_FILENAME = "finetuning_conversations.json" # <--- Output file for fine-tuning data
# *** UPDATED MODEL based on user confirmation and pricing ***
#OPENAI_MODEL = "o4-mini" # Changed from "gpt-3.5-turbo"
OPENAI_MODEL = "gpt-4.1-mini"# --- End Configuration ---

# --- OpenAI Setup ---
try:
    # Initialize OpenAI client (automatically reads OPENAI_API_KEY from environment)
    client = OpenAI()
    print(f"Using OpenAI model: {OPENAI_MODEL}")
    print("Ensure your API key has access to this model.")
except Exception as e:
    print(f"Error initializing OpenAI client. Is OPENAI_API_KEY set?")
    print(f"Error details: {e}")
    exit(1)

# --- Helper Function for OpenAI Call ---
def get_openai_explanation(tweet_text, actual_label, retries=3, delay=5):
    """
    Sends the tweet and actual label to OpenAI to get an explanation.
    Includes basic retry logic for rate limits.
    """
    # Ensure label description makes sense in the prompt
    label_description = "disaster-related" if actual_label == 1 or str(actual_label).lower() == "disaster-related" else "not disaster-related"

    #system_prompt = """Explain concisely why the following tweet is considered relevant to the specified category when labeled by a person. Don't try to change the user opinion, they are asking why their answer is correct""""
    system_prompt = """Explain the rationale behind a human's decision to label the provided tweet as relevant to the specified category. Assume the provided label is correct for this explanation. Concisely identify specific words, phrases, themes, or sentiments in the tweet that likely led to this label. Connect these elements to the category's definition. Do not evaluate, question, or doubt the label's correctness; only explain the possible reasoning behind it."""

    user_prompt = f"Why is the following tweet considered {label_description}? Tweet: \"{tweet_text}\""

    for attempt in range(retries):
        try:
            # Use the configured OPENAI_MODEL variable here
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5, # Adjust temperature for desired creativity/determinism
                max_tokens=200 # Limit explanation length
            )
            explanation = response.choices[0].message.content.strip()
            if not explanation:
                print(f"Warning: Received empty explanation for tweet: {tweet_text[:50]}...")
                return None # Treat empty explanation as failure
            return explanation
        except RateLimitError:
            if attempt < retries - 1:
                print(f"Rate limit hit. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2 # Exponential backoff
            else:
                print("Rate limit error after multiple retries. Skipping this tweet.")
                return None
        except APIError as e:
             # Specifically check for invalid model or access errors
             if "does not exist" in str(e) or "access" in str(e):
                 print(f"Error: Model '{OPENAI_MODEL}' may not exist or your API key doesn't have access.")
                 print(f"Full API error: {e}")
                 # Exit if the model itself is the problem, no point retrying
                 exit(1)
             else:
                 print(f"OpenAI API error: {e}. Skipping this tweet.")
                 return None
        except Exception as e:
            print(f"An unexpected error occurred during OpenAI call: {e}. Skipping this tweet.")
            return None
    return None


# --- Main Processing Logic ---
all_finetuning_conversations = []
mismatch_count = 0
processed_count = 0

print(f"Loading data from {INPUT_JSON_FILENAME}...")
try:
    with open(INPUT_JSON_FILENAME, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    print(f"Loaded {len(results_data)} results.")
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_JSON_FILENAME}' not found.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{INPUT_JSON_FILENAME}'. Is it valid?")
    exit(1)

print("Processing results to find mismatches and generate explanations...")
for i, result in enumerate(results_data):
    # Check for error entries first
    if result.get("classification") == "error":
        continue

    # Get predicted and actual labels (handle potential type differences)
    predicted_label_str = str(result.get("predicted_label", "")).lower()
    actual_label_str = str(result.get("actual_label", "")).lower()
    tweet_text = result.get("tweet", "")

    # Basic validation
    if not tweet_text or not actual_label_str:
         print(f"Warning: Skipping entry {i+1} due to missing tweet or actual_label.")
         continue

    # Identify mismatches
    if predicted_label_str != actual_label_str:
        mismatch_count += 1
        print(f"\nFound mismatch {mismatch_count} (Entry {i+1}):")
        print(f"  Tweet: {tweet_text[:100]}...") # Print start of tweet
        print(f"  Predicted: '{predicted_label_str}', Actual: '{actual_label_str}'")

        # Get explanation from OpenAI
        print(f"  Querying {OPENAI_MODEL} for explanation...") # Show model name
        explanation = get_openai_explanation(tweet_text, actual_label_str)

        if explanation:
            processed_count += 1
            # Format for fine-tuning dataset
            # Use the actual_label_str for the user prompt description for clarity
            label_description = "disaster-related" if actual_label_str == "1" or actual_label_str == "disaster-related" else "not disaster-related"
            conversation = [
                {
                    "role": "user",
                    "content": f"Why is the following tweet considered {label_description}? Tweet: \"{tweet_text}\""
                },
                {
                    "role": "assistant",
                    "content": explanation
                }
            ]
            all_finetuning_conversations.append(conversation)
            print(f"  Explanation received. Added conversation {processed_count} to dataset.")
        else:
            print(f"  Failed to get explanation for this tweet. Skipping.")
        # Optional: Add a small delay to avoid hitting rate limits too quickly
        time.sleep(0.5) # Sleep for 500ms

print(f"\nProcessing complete.")
print(f"Total results processed: {len(results_data)}")
print(f"Total mismatches found: {mismatch_count}")
print(f"Total conversations generated for fine-tuning: {processed_count}")

# --- Save the formatted data ---
# Save in the format {"conversations": [ [conv1_turn1, conv1_turn2], [conv2_turn1, ... ] ] }
output_data_structure = {"conversations": all_finetuning_conversations}

print(f"Saving formatted conversations to {OUTPUT_JSON_FILENAME}...")
try:
    with open(OUTPUT_JSON_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(output_data_structure, f, indent=4, ensure_ascii=False) # Use indent for readability
    print("Successfully saved fine-tuning data.")
except Exception as e:
    print(f"Error saving output file: {e}")