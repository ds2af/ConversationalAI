import csv
import io
import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from openai import AzureOpenAI
# CSV_FILE = "data/tweets1523.csv"
CSV_FILE = "data/tweets1523.csv"

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

def multi_expert_conversation(client, tweet, initial_classification, initial_confidence, initial_reasoning,
                              system_initiator, system_critic, expert_deployments, expert_thresholds,
                              max_rounds_per_level, re_eval_threshold):
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
            critic_user_message = {"role": "user", "content": (
                f"Review the following classification for the tweet:\n"
                f"Tweet: {tweet}\n"
                f"Classification: {current_classification}\n"
                f"Confidence: {current_confidence}\n"
                f"Reasoning: {current_reasoning}\n"
                f"Expert Level: {level+1}, Round {round_count}.\n"
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
    RE_EVAL_THRESHOLD = 0.5

    # Set expert conversation deployments and thresholds.
    EXPERT_DEPLOYMENTS = ["gpt-4o-2"]# ["gpt-4o-mini", "gpt-4o-2", "gpt-4o-2"]
    EXPERT_THRESHOLDS =[0.7]# [.85, 0.88, 0.9]
    MAX_ROUNDS_PER_LEVEL = 0
    CONVERSATION_THRESHOLD = 0.7

    SYSTEM_INITIATOR = (
    '''I am a researcher interested in studying disaster response events using Graph Neural Networks (GNNs). My goal is to analyze historical events in tweets to better understand the timeliness and effectiveness of responses to various types of disasters, including natural disasters, accidents, and other emergency situations.

Each tweet I receive includes a disaster-related hashtag, but not all tweets are directly relevant to my research objective. I want a model to classify tweets into two categories:
1 - Useful for disaster response analysis (tweets that report actual emergencies, accidents, or crises requiring immediate attention).
0 - Not useful for disaster response analysis (tweets that use disaster-related terms metaphorically or in non-emergency contexts).

Consider these guidelines for classification:
- Tweets reporting actual emergencies, accidents, or crises (such as natural disasters, transportation accidents, industrial incidents, or health emergencies) should be classified as 1.
- Tweets using disaster-related terminology in a metaphorical or non-urgent manner should be classified as 0.
- Factual news reports about any disaster or accident should be classified as 1.
- Jokes, memes, or casual mentions that do not indicate real-life emergencies should be classified as 0.

IMPORTANT ABOUT CONFIDENCE SCORES:
- Provide a confidence score between 0 and 1 that represents how certain you are about your classification.
- A score close to 1 means you are very confident in your decision (whether that decision is 0 or 1).
- A score close to 0.5 means you are uncertain about your classification.
- Avoid using a score close to 0; always express your confidence level between 0.5 and 1.

Return your response in exactly the following format:
1. First line: ONLY "0" or "1" (your classification).
2. Second line: Your confidence score as a decimal between 0.5 and 1 (e.g., 0.85).
3. Third line and beyond: Your reasoning for the classification.
'''
)

    SYSTEM_CRITIC = (
        "You are a discerning critic with expertise in disaster response tweet analysis. "
        "Review the following classification, its confidence, and reasoning. "
        "Consider these specific aspects in your critique:\n"
        "1. Is the tweet actually about a real disaster or emergency situation?\n"
        "2. Does the tweet contain actionable information useful for disaster response?\n"
        "3. Is the confidence level appropriate given the content of the tweet?\n"
        "4. Are there any ambiguities or contextual clues that might have been missed?\n\n"
        "IMPORTANT ABOUT CONFIDENCE SCORES:\n"
        "- A confidence score represents how certain the classifier is about their decision\n"
        "- High confidence (close to 1) means high certainty in the classification (whether 0 or 1)\n"
        "- Low confidence (close to 0) means uncertainty about the classification\n"
        "- Suggest appropriate confidence levels in your critique\n\n"
        "Provide a detailed critique and, if appropriate, suggest an alternative classification along with improved reasoning.\n"
        "Keep your response short."
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

        system_message = {"role": "system", "content": SYSTEM_INITIATOR}
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

            # Check if confidence is below threshold, if so, start the conversation with experts
            if confidence_value < CONVERSATION_THRESHOLD:
                updated_classification, updated_confidence, updated_reasoning, conversation_details = multi_expert_conversation(
                    client=client,
                    tweet=tweet,
                    initial_classification=classification,
                    initial_confidence=confidence,
                    initial_reasoning=reasoning,
                    system_initiator=SYSTEM_INITIATOR,
                    system_critic=SYSTEM_CRITIC,
                    expert_deployments=EXPERT_DEPLOYMENTS,
                    expert_thresholds=EXPERT_THRESHOLDS,
                    max_rounds_per_level=MAX_ROUNDS_PER_LEVEL,
                    re_eval_threshold=RE_EVAL_THRESHOLD
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

    # Save any API errors to a separate file
    if error_log:
        with open(OUTPUT_ERROR_LOG, "w", encoding="utf-8") as error_file:
            for error in error_log:
                error_file.write(error + "\n")


    # Save results as usual
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
