# Sentiment Analysis of Movie Reviews

This project implements a machine learning model to classify the sentiment of movie reviews as either positive or negative.

## Project Overview

The core objective is to demonstrate the application of Natural Language Processing (NLP) techniques for sentiment classification. The solution involves data acquisition, text preprocessing, feature extraction, model training, and evaluation.

## Technologies Used

* **Python**
* **Libraries:** pandas, numpy, scikit-learn, NLTK, matplotlib, seaborn, kagglehub

## Dataset

The model is trained and evaluated using the "IMDB Dataset of 50k Movie Reviews" sourced from Kaggle. This dataset comprises 50,000 movie reviews with corresponding positive or negative sentiment labels.

## Methodology

1.  **Data Acquisition:** The dataset is programmatically downloaded from KaggleHub.
2.  **Text Preprocessing:** Raw movie reviews undergo cleaning, including lowercasing, HTML tag removal, punctuation removal, tokenization, stop word removal, and lemmatization.
3.  **Feature Extraction:** Textual data is converted into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) method.
4.  **Model Training:** A Multinomial Naive Bayes classifier is trained on the prepared features.
5.  **Model Evaluation:** The model's performance is assessed using accuracy, a classification report (precision, recall, F1-score), and a confusion matrix.
6.  **Prediction:** The trained model is then used to predict sentiment on new, unseen review texts.

## Files

* `sentiment_analysis_project.ipynb`: The main Google Colab notebook containing all the Python code for data processing, model training, and evaluation.
