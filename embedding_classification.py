import pandas as pd
import numpy as np
from ast import literal_eval

from openai import OpenAI
import os

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
    model = LogisticRegression(max_iter=200)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Detailed classification report
    print(classification_report(y_test, y_pred))

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

def get_embeddings(): 
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    client = OpenAI(api_key=openai_api_key)

    df = pd.read_csv('dataset_and_ratings.csv')

    # # Pooling All Types
    unified_df_h = df[['dialog_id', 'situation', 'speaker_uttr', 'response_human', 'rating_human']]
    unified_df_c = df[['dialog_id', 'situation', 'speaker_uttr', 'response_chatgpt', 'rating_chatgpt']]
    unified_df_e = df[['dialog_id', 'situation', 'speaker_uttr', 'response_chatgpt_empathy', 'rating_chatgpt_empathy']]

    unified_df_h = unified_df_h.assign(type = 'human')
    unified_df_c = unified_df_c.assign(type = 'chatgpt')
    unified_df_e = unified_df_e.assign(type = 'chatgpt_empathy')

    unified_df_h = unified_df_h.rename(columns={'response_human': 'response', 'rating_human': 'rating'})
    unified_df_c = unified_df_c.rename(columns={'response_chatgpt': 'response', 'rating_chatgpt': 'rating'})
    unified_df_e = unified_df_e.rename(columns={'response_chatgpt_empathy': 'response', 'rating_chatgpt_empathy': 'rating'})

    unified_df = pd.concat([pd.concat([unified_df_h, unified_df_c]), unified_df_e])

    def get_embedding(text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    unified_df["combined"] = (
        "Situation: " + unified_df.situation.str.strip() + "; Speaker Utterance: " + unified_df.speaker_uttr.str.strip() + "; Response: " + unified_df.response.str.strip()
    )

    unified_df['3-small-embedding'] = unified_df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
    unified_df.to_csv('6000-embeddings-new.csv', index=False)

def classify_with_embeddings(): 
    unified_df = pd.read_csv('6000-embeddings.csv')

    unified_df['unified_dialog_id'] = unified_df['dialog_id'] + '_type:' + unified_df['type']

    b_df = pd.read_csv('unified_balanced_dataset.csv')
    balanced_df = unified_df[unified_df['unified_dialog_id'].isin(b_df['unified_dialog_id'])].reset_index(drop=True)

    # Set the 'ID' column of balanced_df as a Categorical with the order defined by b_df
    balanced_df['unified_dialog_id'] = pd.Categorical(balanced_df['unified_dialog_id'], categories=b_df['unified_dialog_id'], ordered=True)

    # Sort balanced_df according to the categorical order of 'ID'
    balanced_df = balanced_df.sort_values(by='unified_dialog_id')
    balanced_df = balanced_df.reset_index(drop=True)

    # # Use this line if not referencing the unified balanced dataset
    # balanced_df = balance_dataset(unified_df, 'rating')

    # Check the new class distribution
    print(balanced_df['rating'].value_counts())
    print("")
    print(balanced_df['type'].value_counts())
    print("")

    feature_df = balanced_df[['3-small-embedding']]
    feature_df = feature_df.rename(columns={'3-small-embedding': 'embedding'})
    feature_df["embedding"] = feature_df.embedding.apply(literal_eval).apply(np.array)  # convert string to array

    rating_df = balanced_df[['rating']]
    ratings = rating_df.values.ravel()

    train_classifiers(list(feature_df.embedding.values), ratings)

def main_menu():
    while True:
        print("\nMain Menu")
        print("1. Option 1: Get the embeddings for the dialogues")
        print("2. Option 2: Classify with the embeddings")
        print("0. Exit")

        choice = input("Enter your choice (0-2): ")

        if choice == '1':
            get_embeddings()
        elif choice == '2':
            classify_with_embeddings()
        elif choice == '0':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
