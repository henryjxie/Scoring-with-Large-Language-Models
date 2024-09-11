import pandas as pd
import numpy as np
import json, csv

import sys
import os
import time

from openai import OpenAI

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def balance_dataset (df, field_name): 

    # Separate classes
    df_rating_1 = df[df[field_name] == 1]
    df_rating_2 = df[df[field_name] == 2]
    df_rating_3 = df[df[field_name] == 3]

    # Downsample majority class
    df_rating_2_downsampled = resample(df_rating_2,
                                       replace=False,  # Without replacement
                                       n_samples=len(df_rating_1),  # Match minority class
                                       random_state=42)  # For reproducibility
    df_rating_3_downsampled = resample(df_rating_3,
                                       replace=False,  # Without replacement
                                       n_samples=len(df_rating_1),  # Match minority class
                                       random_state=42)  # For reproducibility

    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_rating_1, df_rating_2_downsampled, df_rating_3_downsampled])

    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # # Check the new class distribution
    # print(df_balanced[field_name].value_counts())
    # print("")

    return df_balanced

def create_fine_tune_jsonl(input_csv, output_jsonl):
    with open(input_csv, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        fine_tune_data = []

        for row in csv_reader:
            user_content = f"You are given a situation context, a speaker utterance, and a response to the speaker utterance in the situation context. " +\
                "Please score the response on a scale of 1 to 3, where a score of 1 means a bad empathetic response, " +\
                "a score of 2 means an okay empathetic response, and a score of 3 means a good empathetic response.\n" +\
                f"Situation: '{row['situation']}' \n Speaker Utterance: '{row['speaker_uttr']}' \n Response: '{row['response']}'"
            
            assistant_content = str(row['rating'])

            fine_tune_entry = {
                "messages": [
                    {"role": "system", "content": "You are an expert rater of empathy in dialogues. Do not output anything else but the empathy score."},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }

            fine_tune_data.append(fine_tune_entry)

        with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
            for entry in fine_tune_data:
                jsonl_file.write(json.dumps(entry) + '\n')

def create_balanced_datasets():
    print("\nCreating Balanced Train and Test Datasets ...")

    balanced_df = pd.read_csv('unified_balanced_dataset.csv')

    # Split the DataFrame into train and test sets
    train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['rating'])

    train_df.to_csv('unified_balanced_dataset_train.csv')
    test_df.to_csv('unified_balanced_dataset_test.csv')

    print('Done.')

def create_jsonl():
    print("\nCreating Fine-Tune JSonl Files ...")

    input_csv = 'unified_balanced_dataset_train.csv'
    output_jsonl = 'unified_balanced_dataset_train.jsonl'

    create_fine_tune_jsonl(input_csv, output_jsonl)

    print('Done.')

def validate_finetune_data_format():
    print("Validating Data Format ...")

    from collections import defaultdict

    data_path = "unified_balanced_dataset_train.jsonl"
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)

    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

    print("Done")

def fine_tune(): 
    print('Fine-tuning ...')

    openai_api_key = os.environ.get('OPENAI_API_KEY')
    client = OpenAI(api_key=openai_api_key)

    uploaded_file = client.files.create(
       file=open("unified_balanced_dataset_train.jsonl", "rb"),
        purpose="fine-tune"
    )

    # print(uploaded_file)

    file_id = uploaded_file.id

    # uploaded_file_t = client.files.create(
    #     file=open("fine-tuning-json-20-percent.jsonl", "rb"),
    #     purpose="fine-tune"
    # )

    # print(uploaded_file_t)

    # file_id_t = uploaded_file_t.id

    fine_tune_job = client.fine_tuning.jobs.create(
        training_file=file_id, 
        # validation_file=file_id_t, 
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={
            "n_epochs": 4,
            "learning_rate_multiplier": 0.5
        }     
    )

    print(f"Fine-tune job created: {fine_tune_job.id}")

    job_id = fine_tune_job.id
    job_status = client.fine_tuning.jobs.retrieve(job_id)
    file_id = job_status.result_files

    while job_status.status not in ['succeeded', 'failed']:
        print(f"Job status: {job_status.status}")
        time.sleep(60)
        job_status = client.fine_tuning.jobs.retrieve(job_id)
        # content = client.files.content(file_id)
        # print(content)

        # events = client.fine_tuning.jobs.list_events(job_id)
        # for event in events['data']:
        #     if event['type'] == 'metrics':
        #         print(f"Step: {event['data']['step']}")
        #         print(f"Training Loss: {event['data']['train_loss']}")
        #         print(f"Validation Loss: {event['data']['valid_loss']}")
        #         print(f"Full Validation Loss: {event['data']['full_valid_loss']}")
        #         print(f"Training Token Accuracy: {event['data']['train_mean_token_accuracy']}")
        #         print(f"Validation Token Accuracy: {event['data']['valid_mean_token_accuracy']}")
        #         print(f"Full Validation Token Accuracy: {event['data']['full_valid_mean_token_accuracy']}")
        #         print("---")

    if job_status.status == 'succeeded':
        fine_tuned_model = job_status.fine_tuned_model
        print(f"Fine-tuned model ID: {fine_tuned_model}")
    else:
        print("Fine-tuning failed.")

    print("Done")

def analyze_results():
    print("Analyzing results ... ")

    openai_api_key = os.environ.get('OPENAI_API_KEY')
    client = OpenAI(api_key=openai_api_key)

    df = pd.read_csv('unified_balanced_dataset_test.csv')
                     
    system_msg = "You are an expert rater of empathy in dialogues. Do not output anything else but the empathy score."

    # Naive Prompt
    empathy_scoring = "You are given a situation context, a speaker utterance, and a response to the speaker utterance in the situation context. " +\
            "Please score the response on a scale of 1 to 3, where a score of 1 means a bad empathetic response, " +\
            "a score of 2 means an okay empathetic response, and a score of 3 means a good empathetic response."

    # # Naive + 3 Dimensions Prompt
    # empathy_scoring = "You are given a situation context, a speaker utterance, and a response to the speaker utterance in the situation context. " +\
    #         "Please score the response on a scale of 1 to 3, where a score of 1 means a bad empathetic response, " +\
    #         "a score of 2 means an okay empathetic response, and a score of 3 means a good empathetic response." +\
    #         "" +\
    #         "Empathy is the ability to understand and share the feelings of another person. " + \
    #         "It is the ability to put yourself in someone else’s shoes and see the world from their perspective. " + \
    #         "Empathy is a complex skill that involves cognitive, emotional, and compassionate components. " + \
    #         "It involves a deeper level of emotional engagement than cognitive empathy prompting action to alleviate another’s distress or suffering.\n" + \
    #         "" +\
    #         "When scoring, consider that empathy is defined by 3 dimensions: cognitive, affective, and compassionate.\n" +\
    #         "- Cognitive empathy is the ability to understand another person’s thoughts, beliefs, and intentions. It is being able to see the world through their eyes and understand their point of view.\n" +\
    #         "- Affective empathy is the ability to experience the emotions of another person. It is feeling what they are feeling, both positive and negative.\n" +\
    #         "- Compassionate empathy is the ability to not only understand and share another person’s feelings, but also to be moved to help if needed. It involves a deeper level of emotional engagement than cognitive empathy prompting action to alleviate another’s distress or suffering.\n" +\
    #         ""

    # # Naive + 3 Dimensions + 15 Subfactors V2 Prompt
    # empathy_scoring = "You are given a situation context, a speaker utterance, and a response to the speaker utterance in the situation context. " +\
    #         "Please score the response on a scale of 1 to 3, where a score of 1 means a bad empathetic response, " +\
    #         "a score of 2 means an okay empathetic response, and a score of 3 means a good empathetic response." +\
    #         "" +\
    #         "Empathy is the ability to understand and share the feelings of another person. " + \
    #         "It is the ability to put yourself in someone else’s shoes and see the world from their perspective. " + \
    #         "Empathy is a complex skill that involves cognitive, emotional, and compassionate components. " + \
    #         "It involves a deeper level of emotional engagement than cognitive empathy prompting action to alleviate another’s distress or suffering.\n" + \
    #         "" +\
    #         "When scoring, consider that empathy is defined by 3 dimensions: cognitive, affective, and compassionate.\n" +\
    #         "- Cognitive empathy is the ability to understand another person’s thoughts, beliefs, and intentions. It is being able to see the world through their eyes and understand their point of view.\n" +\
    #         "- Affective empathy is the ability to experience the emotions of another person. It is feeling what they are feeling, both positive and negative.\n" +\
    #         "- Compassionate empathy is the ability to not only understand and share another person’s feelings, but also to be moved to help if needed. It involves a deeper level of emotional engagement than cognitive empathy prompting action to alleviate another’s distress or suffering.\n" +\
    #         "" +\
    #         "These three empathy dimensions can be further refined into five subfactors each as follows: " + \
    #         "Cognitive Empathy:\n" + \
    #         "- Perspective-Taking: This subfactor measures the ability to mentally adopt another person's viewpoint, understanding how they perceive a situation, including their thoughts, beliefs, and values. It allows you to imagine their experience and predict their reactions.\n" + \
    #         "- Recognition of Emotions: This subfactor involves identifying and understanding the emotions others are experiencing. Beyond facial expressions, it’s about grasping the underlying emotional state, essential for empathetic and appropriate responses.\n" + \
    #         "- Contextual Awareness: This subfactor assesses the ability to consider situational factors that shape someone's thoughts and feelings. It requires understanding the broader context, including environment and cultural background, to respond empathetically.\n" + \
    #         "- Acknowledgment of Speaker's Experience: This subfactor focuses on recognizing and validating the experiences of others. It involves actively listening and showing respect for their feelings and thoughts, which builds trust and emotional connection.\n" + \
    #         "- Clarity of Response: This subfactor evaluates how clearly and accurately you communicate your understanding of another's thoughts and feelings. It ensures your words, tone, and body language effectively convey your empathy.\n" + \
    #         "Affective Empathy:\n" + \
    #         "- Warmth in Tone: This measures how a person's communication conveys friendliness, kindness, and genuine concern. Warmth in tone creates a safe and valued environment, fostering connection and making others feel comforted and supported.\n" + \
    #         "- Sympathetic Responses/Expression of Sympathy: This evaluates the ability to recognize and respond to another's distress with sympathy. It involves acknowledging their feelings and expressing a desire to alleviate their discomfort, showing understanding and offering support.\n" + \
    #         "- Emotional Mirroring: This assesses the ability to reflect another person's emotions by picking up on emotional cues and responding similarly. Emotional mirroring shows that their feelings are understood, fostering a deeper emotional connection.\n" + \
    #         "- Validation of Feelings: This measures how well a person acknowledges and affirms others' emotions as valid. Validation reassures others that their feelings are understood and reasonable, helping them feel seen and heard during emotional distress.\n" + \
    #         "- Emotional Resonance: This assesses the capacity to deeply connect with another's emotions, feeling them to some extent oneself. Emotional resonance creates a shared emotional experience, reinforcing a strong emotional bond.\n" + \
    #         "Compassionate Empathy:\n" + \
    #         "- Encouragement: This subfactor assesses the ability to boost others' morale through positive reinforcement. It involves recognizing efforts and emotions, helping others feel valued and motivated by expressing belief in their abilities.\n" + \
    #         "- Reassurance: This subfactor focuses on comforting others during stress or uncertainty. It involves offering words or actions that alleviate worry, providing a sense of safety." + \
    #         "- Offering Help: This subfactor evaluates the proactive willingness to assist others. It involves recognizing when someone needs support and extending help, either through direct action or providing resourcesn.\n" + \
    #         "- Empowering: This subfactor measures the ability to uplift others by fostering autonomy and self-confidence. Empowering involves encouraging independence and belief in one's capabilities.\n" + \
    #         "- Assistance: This subfactor assesses active involvement in helping others achieve their goals. Assistance is about providing practical support, whether through guidance, completing tasks, or sharing information.\n " + \
    #         ""

    # # Naive + 3 Dimensions + 15 Subfactors V1 Prompt
    # empathy_scoring = "You are given a situation context, a speaker utterance, and a response to the speaker utterance in the situation context. " +\
    #         "Please score the response on a scale of 1 to 3, where a score of 1 means a bad empathetic response, " +\
    #         "a score of 2 means an okay empathetic response, and a score of 3 means a good empathetic response." +\
    #         "" +\
    #         "Empathy is the ability to understand and share the feelings of another person. " + \
    #         "It is the ability to put yourself in someone else’s shoes and see the world from their perspective. " + \
    #         "Empathy is a complex skill that involves cognitive, emotional, and compassionate components. " + \
    #         "It involves a deeper level of emotional engagement than cognitive empathy prompting action to alleviate another’s distress or suffering.\n" + \
    #         "" +\
    #         "When scoring, consider that empathy is defined by 3 dimensions: cognitive, affective, and compassionate.\n" +\
    #         "- Cognitive empathy is the ability to understand another person’s thoughts, beliefs, and intentions. It is being able to see the world through their eyes and understand their point of view.\n" +\
    #         "- Affective empathy is the ability to experience the emotions of another person. It is feeling what they are feeling, both positive and negative.\n" +\
    #         "- Compassionate empathy is the ability to not only understand and share another person’s feelings, but also to be moved to help if needed. It involves a deeper level of emotional engagement than cognitive empathy prompting action to alleviate another’s distress or suffering.\n" +\
    #         "" +\
    #         "These three empathy dimensions can be further refined into five subfactors each as follows: " + \
    #         "\n" + \
    #         "Cognitive Empathy:\n" + \
    #         "- Perspective-Taking: Seeing the world from another person’s viewpoint.\n" + \
    #         "- Recognition of Thoughts: Acknowledging and understanding another person’s thoughts.\n" + \
    #         "- Understanding Intentions: Grasping the reasons behind someone’s actions.\n" + \
    #         "- Contextual Understanding: Understanding the broader context of someone’s situation.\n" + \
    #         "- Inference Accuracy: Accurately inferring another person’s mental states.\n" + \
    #         "\n" + \
    #         "Affective Empathy:" + \
    #         "- Emotional Resonance: Sharing and resonating with another person’s emotions.\n" + \
    #         "- Emotional Matching: Reflecting and mirroring another person’s emotional state.\n" + \
    #         "- Emotional Response: Reacting appropriately to another person’s emotions.\n" + \
    #         "- Emotional Identification: Identifying specific emotions another person is feeling.\n" + \
    #         "- Empathic Concern: Feeling concern and compassion for another’s emotional state.\n" + \
    #         "\n" + \
    #         "Compassionate Empathy:\n" + \
    #         "- Emotional Concern: Feeling concern for another person’s well-being.\n" + \
    #         "- Motivation to Help: Desire to assist someone in need.\n" + \
    #         "- Supportive Actions: Taking concrete steps to help another person.\n" + \
    #         "- Empathic Responsiveness: Responding in an emotionally supportive manner.\n" + \
    #         "- Practical Assistance: Providing tangible help to address the person’s needs.\n" + \
    #         ""

    def scoreEmpathy(context, utterance, response_type):
        completion = client.chat.completions.create(
            # model="ft:gpt-4o-mini-2024-07-18:personal::A2T46n7I", # Epoc: 3; Learning_rate_factor: 1.8
            # model="ft:gpt-4o-mini-2024-07-18:personal::A2TlN5a6", # Epoc: 4; Learning_rate_factor: 1
            # model="ft:gpt-4o-mini-2024-07-18:personal::A2W8Awwg", # Epoc: 4; Learning_rate_factor: 0.75
            model="ft:gpt-4o-mini-2024-07-18:personal::A2Nuqfc5", # Epoc: 4; Learning_rate_factor: 0.5
            # model="ft:gpt-4o-mini-2024-07-18:personal::A2USfVi9", # Epoc: 4; Learning_rate_factor: 0.25
            # model="gpt-4o-mini",
            # model="gpt-4o",
            # model="gpt-4",
            # model="gpt-3.5-turbo",
            temperature=0, 
            top_p = 0.1, 

            messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"{empathy_scoring} \n Situation: '{context}' \n Speaker Utterance: '{utterance}' \n Response: '{response_type}' "}
            ]
        )

        score = completion.choices[0].message.content
        score = score.replace("Empathy Score: ", "")
        score = score.replace("Score: ", "")
        return score

    f = open('unified_balanced_dataset_test_scores.txt', 'w')

    predictions = []
    references = []

    for index, row in df.iterrows():
        if index % 10 == 0:
            print(index)
        dialog_id = df["dialog_id"].iloc[index]
        dialogue_context = df["situation"].iloc[index]
        speaker_uttr = df["speaker_uttr"].iloc[index]
        response = df["response"].iloc[index]
        type = df["type"].iloc[index]
        rating = df["rating"].iloc[index]

        pred_rating = scoreEmpathy(dialogue_context, speaker_uttr, response)
        score = [f'{dialog_id}_type:{type}', pred_rating]
        score = ','.join(score)
        f.write(score + '\n')

        predictions.append(int(pred_rating))
        references.append(rating)

    f.close()

    # Evaluate the model
    accuracy = accuracy_score(predictions, references)
    print(f"Accuracy: {accuracy}")

    print("Done.")

def main_menu():
    while True:
        print("\nMain Menu")
        print("1. Option 1: Create balanced train and test datasets")
        print("2. Option 2: Create fine-tune jsonl files")
        print("3. Option 3: Validate fine-tune data format")
        print("4. Option 4: Fine-tune gpt-to-mini")
        print("5. Option 5: Analyze results")
        print("0. Exit")

        choice = input("Enter your choice (0-5): ")

        if choice == '1':
            create_balanced_datasets()
        elif choice == '2':
            create_jsonl()
        elif choice == '3':
            validate_finetune_data_format()
        elif choice == '4':
            fine_tune()
        elif choice == '5':
            analyze_results()
        elif choice == '0':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()