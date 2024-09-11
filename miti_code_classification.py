import pandas as pd
import numpy as np

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

def generate_miti_array (df): 

    miti_codes_array = np.empty((0, 15)) 

    for index, row in df.iterrows():
        miti_codes = np.zeros((15))

        miti_codes_str = row["miti"].strip("[]").replace("'", "").split(", ")
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

    # print(miti_codes_array)

    return miti_codes_array, df["rating"].values

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
    # print(classification_report(y_test, y_pred))

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

def classify_each_type(): 
    print("Human Responses ----")
    df_h = df[df['type'] == 'human']
    m_c_array_h, ratings_h = generate_miti_array(df_h) 
    train_classifiers (m_c_array_h, ratings_h)

    print("ChatGPT Responses ----")
    df_c = df[df['type'] == 'chatgpt']
    m_c_array_c, ratings_c = generate_miti_array(df_c) 
    train_classifiers (m_c_array_c, ratings_c)

    print("ChatGPT Empathy Responses ----")
    df_e = df[df['type'] == 'chatgpt_empathy']
    m_c_array_e, ratings_e = generate_miti_array(df_e) 
    train_classifiers (m_c_array_e, ratings_e)

def classify_all_types(): 
    print("All Reponses ----")
    m_c_array, ratings = generate_miti_array(df) 
    train_classifiers (m_c_array, ratings)

def main_menu():
    while True:
        print("\nMain Menu")
        print("1. Option 1: Classify with MITI code of 3 types of reponses: Human, ChatGPT, and ChatGPT Empathy, respectively")
        print("2. Option 2: Classify with MITI code of all reponses together")
        print("0. Exit")

        choice = input("Enter your choice (0-2): ")

        if choice == '1':
            classify_each_type()
        elif choice == '2':
            classify_all_types()
        elif choice == '0':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

df = pd.read_csv('unified_balanced_dataset.csv')

if __name__ == "__main__":
    main_menu()
