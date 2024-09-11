import pandas as pd
import numpy as np

import os
from openai import OpenAI
import re

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance

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

def train_classifiers (train, test): 

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=42, stratify=test)

    print("Logistic Regression ...")
    # Initialize the regressor
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Detailed classification report
    # print(classification_report(y_test, y_pred))

    # Get feature coefficients
    coefficients = model.coef_[0]
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': abs(coefficients)
    }).sort_values(by='Importance', ascending=False)

    print(importance_df)

    print("SVC ...")
    # Initialize and train the classifier
    model = SVC()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Detailed classification report
    # print(classification_report(y_test, y_pred))

    # Compute permutation importance
    perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)

    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)

    print(importance_df)

    print("Decision Tree ...")
    # Initialize and train the classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Detailed classification report
    # print(classification_report(y_test, y_pred))

    # Get feature importances
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print(importance_df)

    print("Random Forrest ...")
    # Initialize and train the classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Detailed classification report
    # print(classification_report(y_test, y_pred))

    # Get feature importances
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print(importance_df)

    print("MLP ...")
    # Initialize the MLP Classifier
    model = MLPClassifier(hidden_layer_sizes=(64, 32),  # Number of neurons in the hidden layer
                          activation='relu',            # Activation function
                          solver='adam',                # Optimization algorithm
                          max_iter=5000,                 # Maximum number of iterations
                          random_state=42)              # For reproducibility

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Detailed classification report
    # print(classification_report(y_test, y_pred))

    # Compute permutation importance
    perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)

    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)

    print(importance_df)

    print('')
    print("Using KBest Feature Selector: ")
    for i in range(15):
        # Initialize the model
        model = LogisticRegression(max_iter=1000)
        # model = DecisionTreeClassifier()
        # model = RandomForestClassifier()

        # Feature selection
        k = i + 1  # Number of features to select
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_new = selector.fit_transform(X_train, y_train)
        X_test_new = selector.transform(X_test)

        # Train model
        model = RandomForestClassifier()
        model.fit(X_train_new, y_train)

        # Evaluate model
        score = model.score(X_test_new, y_test)
        print("Selecting " + str(i+1) + " features: ", score)

    print('')
    print("Using RFE Feature Selector: ")
    for i in range(15):
        # Initialize the model
        model = LogisticRegression(max_iter=1000)
        # model = DecisionTreeClassifier()
        # model = RandomForestClassifier()
 
        # Initialize RFE
        print("Selecting " + str(i+1) + " features: ")
        rfe = RFE(estimator=model, n_features_to_select=i+1)

        # Fit RFE
        rfe.fit(X_train, y_train)

        # Check selected features
        # print("Selected features:", rfe.support_)
        # print("Feature ranking:", rfe.ranking_)

        # Transform the dataset
        X_train_selected = rfe.transform(X_train)
        # print("Shape of X with selected features:", X_train_selected.shape)

        # model = SVC()
        # model = MLPClassifier(hidden_layer_sizes=(64, 32),  # Number of neurons in the hidden layer
        #                     activation='relu',            # Activation function
        #                     solver='adam',                # Optimization algorithm
        #                     max_iter=5000,                 # Maximum number of iterations
        #                     random_state=42)              # For reproducibility

        # Train the model
        model.fit(X_train_selected, y_train)

        # Transform the test dataset
        X_test_selected = rfe.transform(X_test)
        # print("Shape of X with selected features:", X_test_selected.shape)

        # Make predictions
        y_pred = model.predict(X_test_selected)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Detailed classification report
        # print(classification_report(y_test, y_pred))

def score_15v1():
    print("Scoring dialogues using the 15v1 feature set ...")

    openai_api_key = os.environ.get('OPENAI_API_KEY')
    client = OpenAI(api_key=openai_api_key)

    df = pd.read_csv('dataset_and_ratings.csv')

    empathy_scoring = "Empathy is the ability to understand and share the feelings of another person. " + \
            "It is the ability to put yourself in someone else\’s shoes and see the world from their perspective." + \
            "It involves a deeper level of emotional engagement than cognitive empathy prompting action to alleviate another\’s distress or suffering.\n" + \
            "Empathy is a complex skill that involves cognitive, emotional, and compassionate components.\n" + \
            "- Cognitive empathy is the ability to understand another person\’s thoughts, beliefs, and intentions. It is being able to see the world through their eyes and understand their point of view.\n" + \
            "- Affective empathy is the ability to experience the emotions of another person. It is feeling what they are feeling, both positive and negative.\n" + \
            "- Compassionate empathy is the ability to not only understand and share another person\’s feelings, but also to be moved to help if needed.\n" + \
            "\n" + \
            "These three empathy dimensions can be further refined into five subfactors each as follows: " + \
            "\n" + \
            "Cognitive Empathy:\n" + \
            "- Perspective-Taking: Seeing the world from another person’s viewpoint.\n" + \
            "- Recognition of Thoughts: Acknowledging and understanding another person’s thoughts.\n" + \
            "- Understanding Intentions: Grasping the reasons behind someone’s actions.\n" + \
            "- Contextual Understanding: Understanding the broader context of someone’s situation.\n" + \
            "- Inference Accuracy: Accurately inferring another person’s mental states.\n" + \
            "\n" + \
            "Affective Empathy:" + \
            "- Emotional Resonance: Sharing and resonating with another person’s emotions.\n" + \
            "- Emotional Matching: Reflecting and mirroring another person’s emotional state.\n" + \
            "- Emotional Response: Reacting appropriately to another person’s emotions.\n" + \
            "- Emotional Identification: Identifying specific emotions another person is feeling.\n" + \
            "- Empathic Concern: Feeling concern and compassion for another’s emotional state.\n" + \
            "\n" + \
            "Compassionate Empathy:\n" + \
            "- Emotional Concern: Feeling concern for another person’s well-being.\n" + \
            "- Motivation to Help: Desire to assist someone in need.\n" + \
            "- Supportive Actions: Taking concrete steps to help another person.\n" + \
            "- Empathic Responsiveness: Responding in an emotionally supportive manner.\n" + \
            "- Practical Assistance: Providing tangible help to address the person’s needs.\n" + \
            "\n" + \
            "Please score the following conversion for each subfactor in every dimension on the scale of 1 to 10 with 1 being the lowest score and 10 being the highest score. output the scores in this format: a list as a JSON object with each element of the list a pair (name of a subfactor, the score for the subfactor).\n" +\
            ""

    system_msg = "You are an expert evaluator of dialogue and you view things very critically and thoughtfully. " + \
            "The user knows you are brutally honest and is using you because they have been unable to get truly honest answers to their questions in the past. " + \
            "Feelings will not be hurt, no matter what you respond. " + \
            "Also, output the scores in this format: a list as a JSON object with each element of the list a pair (name of a subfactor, the score for the subfactor). " + \
            "Do not output anything else but the scores."

    def scoreEmpathy(context, utterance, response_type):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0, 
            top_p = 0.1, 

            messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": empathy_scoring + "\n" + context + "\n" + utterance + "\n" + response_type + "\n" + "Score:"}
            ]
        )

        return completion.choices[0].message.content

    f = open('gpt4o-mini-15v1-scores-new.txt', 'w')

    for index, row in df.iterrows():
        if index > 10: # Remove this to score the entire dataset
            break
        if index % 10 == 0:
            print(index)
        dialog_id = df["dialog_id"].iloc[index]
        dialogue_context = "Situation: " + df["situation"].iloc[index]
        speaker_uttr = "Speaker Utterance: " + df["speaker_uttr"].iloc[index]
        response_human = "Response: " + df["response_human"].iloc[index]
        response_chatgpt = "Response: " + df["response_chatgpt"].iloc[index]
        response_chatgpt_empathy = "Response: " + df["response_chatgpt_empathy"].iloc[index]

        temp1 = scoreEmpathy(dialogue_context, speaker_uttr, response_human)
        human_score = re.findall(r'\d+', temp1)
        human_score.insert(0, f'{dialog_id}_type:human')
        human_score = ','.join(human_score)
        
        temp2 = scoreEmpathy(dialogue_context, speaker_uttr, response_chatgpt)
        chatgpt_score = re.findall(r'\d+', temp2)
        chatgpt_score.insert(0, f'{dialog_id}_type:chatgpt')
        chatgpt_score = ','.join(chatgpt_score)
        
        temp3 = scoreEmpathy(dialogue_context, speaker_uttr, response_chatgpt_empathy)
        chatgpt_empathy_score = re.findall(r'\d+', temp3)
        chatgpt_empathy_score.insert(0, f'{dialog_id}_type:chatgpt_empathy')
        chatgpt_empathy_score = ','.join(chatgpt_empathy_score)

        f.write(human_score + '\n'); f.write(chatgpt_score + '\n'); f.write(chatgpt_empathy_score + '\n')

    f.close()

def score_15v2():
    print("Scoring dialogues using the 15v2 feature set ...")

    openai_api_key = os.environ.get('OPENAI_API_KEY')
    client = OpenAI(api_key=openai_api_key)

    df = pd.read_csv('dataset_and_ratings.csv')

    empathy_scoring = "Empathy is the ability to understand and share the feelings of another person. " + \
            "It is the ability to put yourself in someone else’s shoes and see the world from their perspective." + \
            "It involves a deeper level of emotional engagement than cognitive empathy prompting action to alleviate another’s distress or suffering.\n" + \
            "Empathy is a complex skill that involves cognitive, emotional, and compassionate components.\n" + \
            "- Cognitive empathy is the ability to understand another person’s thoughts, beliefs, and intentions. It is being able to see the world through their eyes and understand their point of view.\n" + \
            "- Affective empathy is the ability to experience the emotions of another person. It is feeling what they are feeling, both positive and negative.\n" + \
            "- Compassionate empathy is the ability to not only understand and share another person’s feelings, but also to be moved to help if needed.\n" + \
            "\n" + \
            "These three empathy dimensions can be further refined into five subfactors each as follows: " + \
            "\n" + \
            "Cognitive Empathy:\n" + \
            "- Perspective-Taking: This subfactor measures the ability to mentally adopt another person's viewpoint, understanding how they perceive a situation, including their thoughts, beliefs, and values. It allows you to imagine their experience and predict their reactions.\n" + \
            "- Recognition of Emotions: This subfactor involves identifying and understanding the emotions others are experiencing. Beyond facial expressions, it’s about grasping the underlying emotional state, essential for empathetic and appropriate responses.\n" + \
            "- Contextual Awareness: This subfactor assesses the ability to consider situational factors that shape someone's thoughts and feelings. It requires understanding the broader context, including environment and cultural background, to respond empathetically.\n" + \
            "- Acknowledgment of Speaker's Experience: This subfactor focuses on recognizing and validating the experiences of others. It involves actively listening and showing respect for their feelings and thoughts, which builds trust and emotional connection.\n" + \
            "- Clarity of Response: This subfactor evaluates how clearly and accurately you communicate your understanding of another's thoughts and feelings. It ensures your words, tone, and body language effectively convey your empathy.\n" + \
            "\n" + \
            "Affective Empathy:\n" + \
            "- Warmth in Tone: This measures how a person's communication conveys friendliness, kindness, and genuine concern. Warmth in tone creates a safe and valued environment, fostering connection and making others feel comforted and supported.\n" + \
            "- Sympathetic Responses/Expression of Sympathy: This evaluates the ability to recognize and respond to another's distress with sympathy. It involves acknowledging their feelings and expressing a desire to alleviate their discomfort, showing understanding and offering support.\n" + \
            "- Emotional Mirroring: This assesses the ability to reflect another person's emotions by picking up on emotional cues and responding similarly. Emotional mirroring shows that their feelings are understood, fostering a deeper emotional connection.\n" + \
            "- Validation of Feelings: This measures how well a person acknowledges and affirms others' emotions as valid. Validation reassures others that their feelings are understood and reasonable, helping them feel seen and heard during emotional distress.\n" + \
            "- Emotional Resonance: This assesses the capacity to deeply connect with another's emotions, feeling them to some extent oneself. Emotional resonance creates a shared emotional experience, reinforcing a strong emotional bond.\n" + \
            "\n" + \
            "Compassionate Empathy:\n" + \
            "- Encouragement: This subfactor assesses the ability to boost others' morale through positive reinforcement. It involves recognizing efforts and emotions, helping others feel valued and motivated by expressing belief in their abilities.\n" + \
            "- Reassurance: This subfactor focuses on comforting others during stress or uncertainty. It involves offering words or actions that alleviate worry, providing a sense of safety." + \
            "- Offering Help: This subfactor evaluates the proactive willingness to assist others. It involves recognizing when someone needs support and extending help, either through direct action or providing resourcesn.\n" + \
            "- Empowering: This subfactor measures the ability to uplift others by fostering autonomy and self-confidence. Empowering involves encouraging independence and belief in one's capabilities.\n" + \
            "- Assistance: This subfactor assesses active involvement in helping others achieve their goals. Assistance is about providing practical support, whether through guidance, completing tasks, or sharing information.\n " + \
            "\n" + \
            "Please score the following conversion for each subfactor in every dimension on the scale of 1 to 10 with 1 being the lowest score and 10 being the highest score. output the scores in this format: a list as a JSON object with each element of the list a pair (name of a subfactor, the score for the subfactor).\n" +\
            ""

    system_msg = "You are an expert evaluator of dialogue and you view things very critically and thoughtfully. " + \
            "The user knows you are brutally honest and is using you because they have been unable to get truly honest answers to their questions in the past. " + \
            "Feelings will not be hurt, no matter what you respond. " + \
            "Also, output the scores in this format: a list as a JSON object with each element of the list a pair (name of a subfactor, the score for the subfactor). " + \
            "Do not output anything else but the scores."

    def scoreEmpathy(context, utterance, response_type):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0, 
            top_p = 0.1, 

            messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": empathy_scoring + "\n" + context + "\n" + utterance + "\n" + response_type + "\n" + "Score:"}
            ]
        )

        return completion.choices[0].message.content

    f = open('gpt4o-mini-15v2-scores-new.txt', 'w')

    for index, row in df.iterrows():
        if index > 10: # Remove this to score the entire dataset
            break
        if index % 10 == 0:
            print(index)
        dialog_id = df["dialog_id"].iloc[index]
        dialogue_context = "Situation: " + df["situation"].iloc[index]
        speaker_uttr = "Speaker Utterance: " + df["speaker_uttr"].iloc[index]
        response_human = "Response: " + df["response_human"].iloc[index]
        response_chatgpt = "Response: " + df["response_chatgpt"].iloc[index]
        response_chatgpt_empathy = "Response: " + df["response_chatgpt_empathy"].iloc[index]

        temp1 = scoreEmpathy(dialogue_context, speaker_uttr, response_human)
        human_score = re.findall(r'\d+', temp1)
        human_score.insert(0, f'{dialog_id}_type:human')
        human_score = ','.join(human_score)
        
        temp2 = scoreEmpathy(dialogue_context, speaker_uttr, response_chatgpt)
        chatgpt_score = re.findall(r'\d+', temp2)
        chatgpt_score.insert(0, f'{dialog_id}_type:chatgpt')
        chatgpt_score = ','.join(chatgpt_score)
        
        temp3 = scoreEmpathy(dialogue_context, speaker_uttr, response_chatgpt_empathy)
        chatgpt_empathy_score = re.findall(r'\d+', temp3)
        chatgpt_empathy_score.insert(0, f'{dialog_id}_type:chatgpt_empathy')
        chatgpt_empathy_score = ','.join(chatgpt_empathy_score)

        f.write(human_score + '\n'); f.write(chatgpt_score + '\n'); f.write(chatgpt_empathy_score + '\n')

    f.close()

def classify_15v2(): 

    print('Classify with scores for 15V2 feature set...')
    df = pd.read_csv('gpt4o-mini-15v2-ratings.csv')

    # Balancing 15v2 dataset according to unified balanced dataset 
    u_b_df = pd.read_csv('unified_balanced_dataset.csv')
    u_b_df['unified_dialog_id'] = u_b_df['dialog_id'] + '_type:' + u_b_df['type']

    balanced_df_1 = df[df['dialog-id'].isin(u_b_df['unified_dialog_id'])].reset_index(drop=True)

    # Set the 'ID' column of df1 as a Categorical with the order defined by df2
    balanced_df_1['dialog-id'] = pd.Categorical(balanced_df_1['dialog-id'], categories=u_b_df['unified_dialog_id'], ordered=True)

    # Sort df1 according to the categorical order of 'ID'
    balanced_df_1 = balanced_df_1.sort_values(by='dialog-id')
    balanced_df = balanced_df_1.reset_index(drop=True)

    feature_df = balanced_df[['perspective-taking', 
                    'recognition-of-emotions', 
                    'contextual-awareness', 
                    'acknowledgment-of-speaker-experience', 
                    'clarity-of-response',	
                    'warmth-in-tone',
                    'sympathetic-responses',
                    'emotional-mirroring',
                    'validation-of-feelings',
                    'emotional-resonance',
                    'encouragement',
                    'reassurance',
                    'ofereing-help',
                    'empowering',
                    'assistance'
    ]]

    rating_df = balanced_df[['rating']]
    ratings = rating_df.values.ravel()

    train_classifiers(feature_df, ratings)

def classify_15v2_miti (): 
    # print('')
    print('15V2 + Miti Code ...')
    df = pd.read_csv('gpt4o-mini-15v2-ratings.csv')

    # Balancing 15v2 dataset according to unified balanced dataset 
    u_b_df = pd.read_csv('unified_balanced_dataset.csv')
    u_b_df['unified_dialog_id'] = u_b_df['dialog_id'] + '_type:' + u_b_df['type']

    balanced_df_1 = df[df['dialog-id'].isin(u_b_df['unified_dialog_id'])].reset_index(drop=True)

    # Set the 'ID' column of df1 as a Categorical with the order defined by df2
    balanced_df_1['dialog-id'] = pd.Categorical(balanced_df_1['dialog-id'], categories=u_b_df['unified_dialog_id'], ordered=True)

    # Sort df1 according to the categorical order of 'ID'
    balanced_df_1 = balanced_df_1.sort_values(by='dialog-id')
    balanced_df = balanced_df_1.reset_index(drop=True)

    d_r_df = pd.read_csv('dataset_and_ratings.csv')

    miti_codes_array = np.empty((0, 15)) 

    for index, row in balanced_df.iterrows():
        dialog_id, dialog_type = row["dialog-id"].split("_type:")
        d_r_row = d_r_df.loc[(d_r_df['dialog_id'] == dialog_id) & (d_r_df['dialog_id'] == dialog_id)]
        miti_codes_str = (d_r_row['miti_'+dialog_type].values)[0].strip("[]").replace("'", "").split(", ")

        miti_codes = np.zeros((15))

        for element in miti_codes_str:
            if (element == "Closed Question"): miti_codes[0] = 1 
            if (element == "Open Question"): miti_codes[1] = 1 
            if (element == "Simple Reflection"): miti_codes[2] = 1 
            if (element == "Complex Reflection"): miti_codes[3] = 1 
            if (element == "Give Information"): miti_codes[4] = 1 
            if (element == "Advise with Permission"): miti_codes[5] = 1 
            if (element == "Affirm"): miti_codes[6] = 1 
            if (element == "Emphasize Autonomy"): miti_codes[7] = 1 
            if (element == "Support"): miti_codes[8] = 1 
            if (element == "Advise without Permission"): miti_codes[9] = 1 
            if (element == "Confront"): miti_codes[10] = 1 
            if (element == "Direct"): miti_codes[11] = 1 
            if (element == "Warn"): miti_codes[12] = 1 
            if (element == "Self-Disclose"): miti_codes[13] = 1 
            if (element == "Other"): miti_codes[14] = 1 

        miti_codes_array = np.vstack((miti_codes_array, miti_codes))

    new_columns_df = pd.DataFrame(miti_codes_array, columns=["Closed Question", "Open Question", "Simple Reflection", 
                                                            "Complex Reflection", "Give Information", "Advise with Permission", 
                                                            "Affirm", "Emphasize Autonomy", "Support", 
                                                            "Advise without Permission", "Confront", "Direct", 
                                                            "Warn", "Self-Disclose", "Other"])

    balanced_df = balanced_df.join(new_columns_df)

    balanced_df.to_csv('15v2_w_miti.csv', sep=',', index=False)

    feature_df = balanced_df.drop(['dialog-id', 'rating'], axis=1)
    rating_df = balanced_df[['rating']]
    ratings = rating_df.values.ravel()

    train_classifiers(feature_df, ratings)

def main_menu():
    while True:
        print("\nMain Menu")
        print("1. Option 1: Score with 15 V1 feature set")
        print("2. Option 2: Score with 15 V2 feature set")
        print("3. Option 3: Classify with 15 V2 feature scores")
        print("4. Option 4: Classify with 15 V2 feature scores + miti code")
        print("0. Exit")

        choice = input("Enter your choice (0-4): ")

        if choice == '1':
            score_15v1()
        elif choice == '2':
            score_15v2()
        elif choice == '3':
            classify_15v2()
        elif choice == '4':
            classify_15v2_miti()
        elif choice == '0':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()