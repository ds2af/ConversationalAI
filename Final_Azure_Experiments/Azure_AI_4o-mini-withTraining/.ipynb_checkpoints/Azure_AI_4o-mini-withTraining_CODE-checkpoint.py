import csv
import io
import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from openai import AzureOpenAI

# CSV files for tweets and training examples.
CSV_FILE = "data/tweets1523.csv"
TRAINING_CSV = "data/train.csv"
NUM_TRAINING_EXAMPLES = 1000  # Fixed number of training examples to include.

def normalize_classification(classification, reasoning=""):
    """
    Normalize classification to ensure it's either "0" or "1".
    Uses reasoning text to help determine the correct classification if needed.
    """
    cleaned = re.sub(r'[^01]', '', classification.strip())
    if cleaned in ["0", "1"]:
        return cleaned

    lower_reasoning = reasoning.lower()
    # Keywords for non-emergency or non-disaster content.
    negative_keywords = [
        "not disaster", "not a disaster", "non-disaster", "irrelevant", 
        "metaphor", "joke", "not related", "not about disaster", 
        "no emergency", "not an emergency"
    ]
    # Keywords indicating a true emergency (including accidents).
    positive_keywords = [
        "disaster", "emergency", "crisis", "evacuation", 
        "fire", "flood", "earthquake", "hurricane", 
        "accident", "crash", "collision", "explosion", 
        "ambulance", "rescue", "incident"
    ]

    if any(term in lower_reasoning for term in negative_keywords):
        return "0"
    elif any(term in lower_reasoning for term in positive_keywords):
        return "1"
    return "0"

def load_and_preprocess_data(csv_file):
    """
    Load and preprocess the dataset, using the entire dataset without splitting.
    Returns the dataframe and the original dataframe.
    """
    try:
        df = pd.read_csv(csv_file)
        original_df = df.copy()
        return df, original_df
    except Exception as e:
        print(f"Error loading or preprocessing data: {str(e)}")
        return None, None

def load_training_examples(training_csv, num_examples, seed=42):
    """
    Load a fixed number of training examples from the provided CSV by randomly sampling rows.
    Each example should contain a tweet text and its target label.
    
    Parameters:
      training_csv (str): Path to the training CSV file.
      num_examples (int): Number of examples to sample.
      seed (int): Random seed for reproducibility.
      
    Returns:
      str: A formatted string with the selected examples.
    """
    try:
        train_df = pd.read_csv(training_csv)
        # Randomly sample num_examples rows with the given seed.
        train_df = train_df.sample(n=num_examples, random_state=seed)
        examples = []
        for i, row in train_df.iterrows():
            tweet_text = row.get('text', '').strip()
            target = row.get('target', '').strip()
            # Format each example. Adjust the formatting if needed.
            example_str = f"Example {i+1}:\nTweet: {tweet_text}\nTarget: {target}"
            examples.append(example_str)
        return "\n\n".join(examples)
    except Exception as e:
        print(f"Error loading training examples: {str(e)}")
        return ""


def multi_expert_conversation(client, tweet, initial_classification, initial_confidence, initial_reasoning,
                              system_initiator, system_critic, expert_deployments, expert_thresholds,
                              max_rounds_per_level, re_eval_threshold, training_examples):
    """
    Conduct a hierarchical multi-expert conversation.
    Returns:
        final_classification, final_confidence, final_reasoning, conversation_details (structured JSON).
    """
    current_classification = initial_classification
    current_confidence = initial_confidence
    current_reasoning = initial_reasoning
    conversation_details = {
        "initial_response": {
            "classification": initial_classification,
            "confidence": initial_confidence,
            "reasoning": initial_reasoning
        },
        "rounds": []
    }

    for level, deployment in enumerate(expert_deployments):
        threshold = expert_thresholds[level]
        round_count = 0
        while round_count < max_rounds_per_level:
            try:
                if float(current_confidence) >= threshold:
                    conversation_details["summary"] = f"Threshold of {threshold} reached at expert level {level+1} after {round_count} rounds."
                    return current_classification, current_confidence, current_reasoning, conversation_details
            except (ValueError, TypeError):
                pass
            round_count += 1
            round_info = {"expert_level": level + 1, "round": round_count}
            print(f"Talking to Expert Level {level+1}, Round {round_count}: Sending tweet and current classification for critique.")

            critic_system_message = {"role": "system", "content": system_critic}
            # Only include training examples in the first round for context.
            if round_count == 1:
                training_text = "\n\nHere are some example tweets for reference:\n" + training_examples
            else:
                training_text = ""
            critic_user_message = {"role": "user", "content": (
                f"Review the following classification for the tweet:\n"
                f"Tweet: {tweet}\n"
                f"Classification: {current_classification}\n"
                f"Confidence: {current_confidence}\n"
                f"Reasoning: {current_reasoning}\n"
                f"Expert Level: {level+1}, Round {round_count}.\n"
                f"{training_text}\n"
                "Please provide your critique and suggestions."
            )}
            critic_messages = [critic_system_message, critic_user_message]
            critic_response = client.chat.completions.create(
                model=deployment,
                messages=critic_messages,
                max_tokens=300,
                temperature=0.5,
                top_p=0.5
            )
            critic_answer = critic_response.choices[0].message.content.strip()
            round_info["critic_response"] = critic_answer
            print(f"Received from Expert Level {level+1}, Round {round_count}")

            followup_user_message = {"role": "user", "content": (
                f"Expert feedback (Level {level+1}, Round {round_count}): {critic_answer}\n"
                "Please provide an updated classification, confidence, and reasoning in the same format.\n"
                f"If your confidence is less or equal to {re_eval_threshold}, classify it as 0 and re-evaluate the confidence and reasoning."
            )}
            system_message = {"role": "system", "content": system_initiator}
            followup_messages = [system_message, followup_user_message]
            followup_response = client.chat.completions.create(
                model=deployment,
                messages=followup_messages,
                max_tokens=150,
                temperature=0.5,
                top_p=0.5
            )
            followup_answer = followup_response.choices[0].message.content.strip()
            round_info["initiator_followup"] = followup_answer
            print(f"Initiator follow-up response from Expert Level {level+1}, Round {round_count}")
            followup_parts = followup_answer.split('\n')
            if len(followup_parts) >= 3:
                current_classification = followup_parts[0].strip()
                current_confidence = followup_parts[1].strip()
                current_reasoning = '\n'.join(followup_parts[2:]).strip()
            conversation_details["rounds"].append(round_info)
            try:
                if float(current_confidence) >= threshold:
                    conversation_details["summary"] = f"Threshold of {threshold} reached at expert level {level+1} after {round_count} rounds."
                    return current_classification, current_confidence, current_reasoning, conversation_details
            except (ValueError, TypeError):
                pass

        conversation_details.setdefault("escalations", []).append({
            "expert_level": level + 1,
            "message": f"Escalating from expert level {level+1} due to low confidence (current: {current_confidence}, threshold: {threshold})."
        })

    conversation_details["summary"] = "All expert levels exhausted."
    return current_classification, current_confidence, current_reasoning, conversation_details

def main():
    try:
        with open("api_key.txt", "r") as file:
            api_key = file.readline().rstrip()
    except FileNotFoundError:
        print("Error: api_key.txt file not found. Please create this file with your API key.")
        return

    OUTPUT_CSV = "classified_tweets.csv"
    OUTPUT_JSON = "classified_tweets.json"
    OUTPUT_ERROR_LOG = "error_log.txt"
    RE_EVAL_THRESHOLD = 0

    # Set expert conversation deployments and thresholds.
    EXPERT_DEPLOYMENTS =[]#["gpt-4o-2"]
    EXPERT_THRESHOLDS = []#[0.9]
    MAX_ROUNDS_PER_LEVEL = 3  # Set number of rounds per level.
    CONVERSATION_THRESHOLD = 0.9

    SYSTEM_INITIATOR = (
    """
I am a researcher studying disaster events using Graph Neural Networks (GNNs). My objective is to analyze historical tweets to determine if they are truly related to disaster events or not. Based on a thorough review of annotated true labels from our training dataset (as provided in the CSV file), only tweets that report or describe an actual disaster event—such as natural disasters, major accidents, serious incidents, notable criminal cases, industrial incidents, terrorism events, or any significant emergency—should be classified as disaster-related (label “1”). Tweets that employ disaster-related language in a metaphorical, humorous, or non-event context should be classified as not disaster-related (label “0”).

**Classification Guidelines:**

**1 - Disaster-Related:**  
- The tweet must clearly report or describe a verifiable disaster event. This includes on-the-ground accounts, emergency notifications, or detailed news reporting that involve natural disasters, major accidents, serious incidents, notable criminal cases, industrial accidents, terrorism events, or any significant emergency.

**0 - Not Disaster-Related:**  
- The tweet uses disaster-related words figuratively, humorously, or in a non-literal manner. Casual mentions, jokes, memes, or any figurative language that do not correspond to a real event should be classified as “0.”

**IMPORTANT ABOUT CONFIDENCE SCORES:**  
- Provide a confidence score between 0 and 1 indicating your certainty about the classification. A score close to 1 represents high certainty, while a score near 0 indicates significant uncertainty.
- Reference the true labels in the attached CSV file to ensure your decision is consistent with the manual annotations.

Return your response in exactly the following format:
1. First line: ONLY "0" or "1" (your classification).
2. Second line: Your confidence score as a decimal between 0 and 1 (e.g., 0.85).
3. Third line and beyond: Your reasoning for the classification, highlighting specific textual clues or contextual indicators.
    """
)

    SYSTEM_CRITIC = (
    """
You are a discerning critic with expertise in disaster tweet classification. Based on the reviewed true labels from our CSV dataset, evaluate the following classification decision, its confidence score, and the accompanying reasoning.

Consider these specific aspects in your critique:
1. Does the tweet actually report or refer to a verifiable disaster event such as a natural disaster, major accident, serious incident, notable criminal case, industrial incident, terrorism event, or any significant emergency?
2. Is the confidence score appropriate given the clarity and certainty of the tweet’s content?
3. Are there any ambiguities or subtle language cues that might have been overlooked in the classification process?
4. Does the provided reasoning clearly connect the tweet’s language and context to the guidelines derived from the true labels in the dataset?

If necessary, provide an alternative classification with revised reasoning and suggest an adjusted confidence score that better aligns with the annotated true labels.

Keep your critique concise and direct.
    """
)

    conversation_history = []
    confidence_history = []

    endpoint = "https://sv-openai-research-group4.openai.azure.com/"
    initiator_deployment = "gpt-4o-mini"
    api_version = "2024-12-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )

    print("Loading and preprocessing data...")
    df, original_df = load_and_preprocess_data(CSV_FILE)
    if df is None:
        print(f"Error: Could not load data from {CSV_FILE}")
        return

    # Load training examples from the training CSV.
    training_examples = load_training_examples(TRAINING_CSV, NUM_TRAINING_EXAMPLES)
    
    # Build an examples prompt to be appended to the system messages.
    examples_prompt = "\n\nExample tweets for reference:\n" + training_examples

    results = []
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    tweet_count = 0

    print(f"Processing {len(df)} tweets...")

    error_log = []

    for index, row in df.iterrows():
        tweet = row['text']  # Use the raw tweet text without modification
        if 'target' in row:
            true_label = row['target']
            true_labels.append(true_label)
        else:
            true_label = None

        # Append training examples to the SYSTEM_INITIATOR message.
        system_message = {"role": "system", "content": SYSTEM_INITIATOR + examples_prompt}
        user_message = {"role": "user", "content": f"Tweet: {tweet}"}
        messages = [system_message, user_message]

        try:
            response = client.chat.completions.create(
                model=initiator_deployment,
                messages=messages,
                max_tokens=150,
                temperature=0.5,
                top_p=0.5
            )
            answer = response.choices[0].message.content.strip()
            parts = answer.split('\n')
            if len(parts) >= 3:
                classification = parts[0].strip()
                confidence = parts[1].strip()
                reasoning = '\n'.join(parts[2:]).strip()
            else:
                classification = "N/A"
                confidence = "0"
                reasoning = answer

            classification = normalize_classification(classification, reasoning)
            predicted_labels.append(classification)

            try:
                confidence_value = float(confidence)
                confidence_value = max(0.5, min(1.0, confidence_value))
                confidence_scores.append(confidence_value)
            except (ValueError, TypeError):
                confidence_value = 0.5
                confidence_scores.append(confidence_value)

            if true_label is not None:
                try:
                    is_correct = str(true_label) == classification
                    confidence_history.append({'confidence': confidence_value, 'correct': is_correct})
                except Exception as e:
                    pass

            # Check if confidence is below threshold, if so, start the conversation with experts.
            if confidence_value < CONVERSATION_THRESHOLD:
                updated_classification, updated_confidence, updated_reasoning, conversation_details = multi_expert_conversation(
                    client=client,
                    tweet=tweet,
                    initial_classification=classification,
                    initial_confidence=confidence,
                    initial_reasoning=reasoning,
                    system_initiator=SYSTEM_INITIATOR + examples_prompt,
                    system_critic=SYSTEM_CRITIC,
                    expert_deployments=EXPERT_DEPLOYMENTS,
                    expert_thresholds=EXPERT_THRESHOLDS,
                    max_rounds_per_level=MAX_ROUNDS_PER_LEVEL,
                    re_eval_threshold=RE_EVAL_THRESHOLD,
                    training_examples=training_examples
                )

                predicted_labels[-1] = updated_classification
                try:
                    updated_confidence_value = float(updated_confidence)
                    updated_confidence_value = max(0.5, min(1.0, updated_confidence_value))
                    confidence_scores[-1] = updated_confidence_value
                except (ValueError, TypeError):
                    pass

                classification = updated_classification
                confidence = updated_confidence
                reasoning = updated_reasoning

                if true_label is not None:
                    try:
                        is_correct = str(true_label) == updated_classification
                        confidence_history[-1] = {'confidence': updated_confidence_value, 'correct': is_correct}
                    except Exception as e:
                        pass
            else:
                conversation_details = {"initial_response": answer, "rounds": []}

            results.append({
                "tweet": tweet,
                "original_tweet": row['text'],
                "classification": classification,
                "confidence": confidence,
                "reasoning": reasoning,
                "conversation": conversation_details,
                "true_label": true_label
            })
            print(f"Processing tweet {tweet_count + 1}: {tweet[:50]}...")
        except Exception as e:
            error_message = f"Error processing tweet: {tweet[:50]}... \nError: {str(e)}"
            print(error_message)
            error_log.append(error_message)

        tweet_count += 1

    # Save any API errors to a separate file.
    if error_log:
        with open(OUTPUT_ERROR_LOG, "w", encoding="utf-8") as error_file:
            for error in error_log:
                error_file.write(error + "\n")

    # Save results as CSV and JSON.
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_file:
        fieldnames = ["tweet", "original_tweet", "classification", "confidence", "reasoning", "conversation", "true_label"]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    print(f"Classification complete. Results saved to {OUTPUT_CSV} and {OUTPUT_JSON}")
    if error_log:
        print(f"Some API errors were encountered. Details saved to {OUTPUT_ERROR_LOG}")

if __name__ == "__main__":
    main()
