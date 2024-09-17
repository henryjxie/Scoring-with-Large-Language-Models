# Scoring with Large Language Models: A Study on Measuring Empathy of Responses in Dialogue
Henry Xie

October 1, 2024

## Overview
In recent years, Large Language Models (LLMs) have become more powerful than ever, with the ability to complete increasingly complex tasks. One task that Large Language Models have potentials in is scoring, that is, assigning a number from a certain scale to a subject in question. Employing LLMs in scoring tasks is powerful and convenient in that it can help provide a quick measure of the subject. However, because LLMs and LLM scoring are recent topics, the process behind LLM scoring is often obsecure.

In this study, we attempt to understand how LLMs score, specifically how they measure and score empathy about responses in dialogues. We propose methods for modeling the performance of current state-of-the-art LLMs and fine-tuned LLMs when they score dialogue responses on their empathy. We further introduce a set of 15 subfactors combined with the Motivational Interviewing Treatment Integrity (MITI) code provided in the dataset employed in this study to help in our comprehension of how LLMs score and understand empathy. The source code in this repository contains the prototype implementations of the approaches we introduced in our study.

## About this Repository
This repository contains the code for our study on how LLM scoring is utilized in measuring empathy of responses in dialogues.

#### Data files
`dataset_and_ratings.csv`: This CSV file contains the dataset we utilized in this study. It originates from Anuradha Welivita and Pearl Pu's paper "Is ChatGPT More Empathetic than Humans?" This dataset contains 2000 situation-speaker utterance pairs, along with three different responses to the speaker utterance for each of the 2000 pairs. The three different responses are from either a human, ChatGPT, or ChatGPT with empathy defined. This dataset also contains the Motivational Interviewing Treatment Integrity (MITI) code provided for each response. Each of the 2000 situation-speaker utterance pairs have a unique dialogue identifier.

`unified_balanced_dataset.csv `: This CSV file contains the balanced dataset we sourced from the original dataset. This balanced dataset contains the same number situation-speaker utterance-response triplets with a human-rated empathy score of 1, 2, and 3 respectively. In the original dataset, there are an uneven number of situation-speaker utterance-response triplets with a score of 1, 2, and 3 respectively, which significantly affects the results. The triplets with a human-rated score of 1 are fewest at 640. So in this balanced dataset, we keep 640 situation-speaker utterance-response triplets with a human rated score of 1 and them randomly selected 640 situation-speaker utterance-response triplets with a human rated score of 2 and 640 
situation-speaker utterance-response triplets with a human rated score of 3 from the original dataset.

`gpt-4o-mini-15v2-ratings.csv`: This CSV file contains the GPT-4o-mini scores for all 6000 situation-speaker utterance-response triplets, giving a score from 1 to 10 for each of the 15 subfactors of empathy, with a score of 1 meaning that the subfactor is not prevalent in the response and a score of 10 meaning that the subfactor is extremely prevalent in the response.

`6000-embeddings.zip`: This zip file contains the embeddings done by OpenAI's text-embedding-3-small embedding model for all 6000 situation-speaker utterance-response triplets. You will need to unzip this file to view the content.

#### Python Files
`finetune-gpt-4o-mini.py`: This python file, when ran, offers a main menu with five options. The first option is to create the train and test datasets from the unified balanced dataset. The second option is to create the json file of the train dataset for fine-tuning. The third option is to validate the json file created in the second option, to make sure they comply with what OpenAI's fine-tuning facility requires. The fourth option is to fine-tune GPT-4o-mini with the train dataset. The fifth option is to use the fine-tuned model produced in the fourth option to score on the test dataset, either using the Naive prompt, the 3 dimensions prompt, or the 15 V2 subfactors prompt, and analyzing the accuracy of those scores.

`embedding_classifcation.py`: This python file, when ran, offers a main menu with two options. The first option is to use OpenAI's text-embedding-3-small embedding model to embed the 6000 different situation-speaker utterance-response triplets. (To avoid uncessary invocation of the API Embedding API, a pre-generated embeddeding file is provided in 6000-embeddings.zip.) The second option is to train classifier models with the embeddings produced by the first option to predict the empahty scores.

`subfactor_classifcation.py`: This python file, when ran, offers a main menu with four options. The first option is to use GPT-4o-mini to score all 6000 situation-speaker utterance-response triplets with the 15 V1 subfactor set by giving each subfactor a score from 1 to 10. The second option is to score all 6000 situation-speaker utterance-response triplets with the 15 V2 subfactor set by giving each subfactor a score from 1 to 10. (To avoid uncessary invocation of the OpenAI API, a pre-generated score file using 15 V2 subfactors is provided in gpt-4o-mini-15v2-ratings.csv.) The third option is to train the classifier models on the scores produced by the first option. The fourth option is to train the classifier models on the scores produced by the second option.

`miti_code_classification.py`: This python file, when ran, offers two options. The first option is to create the vectors for the MITI code and train classifiers on the MITI codes of the human responses, ChatGPT responses, and ChatGPT with empathy defined responses respectively. The second option is to train classifiers on all the responses together.

## Dependencies
1. You will need to install Python version 3.12.4 on your computer.
2. You will need to import the openai python api package. You can do this by running `pip install openai`.
3. This study requires you to have an OpenAI API key. You will need to store your API key in an environment variable namely `OPENAI_API_KEY`.
