"""
Purpose: Processes and analyzes artist names for music genre classification using
         text preprocessing and various machine learning models.

Key Functions:
    - process_data(): Prepares and processes the dataset for model training and evaluation.
    - run_experiments(): Executes machine learning experiments using EchoNest features and other feature combinations.
    - evaluate_model(): Evaluates the performance of trained models and outputs results.

Notes:
    - This file is a modified version of artistName.py accounting for artist name duplication leading to overfitting
    - EchoNest features are combined with other features like track names and metadata for comprehensive analysis.
    - Some configurations and model setups are commented out for reference and potential future use.
"""

from initialPreprocessing import gen_Train_and_Test, top_tracks
import pandas as pd
from sklearn.model_selection import train_test_split
from svm import svm
from knn import knn
from nb import nb
from sgd import sgd
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string

# Splits the dataset into training and testing sets based on the provided features
# returns split data
def gen_Train_and_Test(data, feature, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(feature, data, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# vectorises the given data through the chosen method (BOW or TF-IDF)
def vectorise(data, method):
    if method == "tfidf":
        vectorizer = TfidfVectorizer()
    elif method == "bow":
        vectorizer = CountVectorizer()
    return vectorizer.fit_transform(data)

# Executes machine learning experiments using the processed data
def run_mega(sample, processed ,descText):
    X_train, X_test, y_train, y_test = gen_Train_and_Test(genres, processed)

    # testing with found optimal k values
    # knn(X_train, X_test, y_train, y_test,descText,1)
    # knn(X_train, X_test, y_train, y_test,descText,3)
    # knn(X_train, X_test, y_train, y_test,descText,5)
    # knn(X_train, X_test, y_train, y_test,descText,7)
    # knn(X_train, X_test, y_train, y_test,descText,9)
    # knn(X_train, X_test, y_train, y_test,descText,21)
    # knn(X_train, X_test, y_train, y_test,descText,99)
    # knn(X_train, X_test, y_train, y_test,descText,199)
    
    # nb(X_train.toarray(), X_test.toarray(), y_train, y_test,descText)
    sgd(X_train, X_test, y_train, y_test,descText) # prints accuracy scores on completion

# Processes artist names by optionally coverting to lowercase, removing punctuation and stop words, and stemming (reducing words to root form)
# returns a list of processed artist names  
def process_artist_names(names, lowercase=False,use_stop_words=False, remove_punctuation=False):
    stemmer = PorterStemmer()

    if use_stop_words:
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set()

    processed_names = []

    for name in names:
        thisname = name
        if lowercase:
            thisname = thisname.lower()
        tokens = word_tokenize(thisname)

        if remove_punctuation:
            processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token not in string.punctuation]
        else:
            processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

        processed_name = ' '.join(processed_tokens)
        processed_names.append(processed_name)

    return processed_names

if __name__ == "__main__":                                                       
    sample = top_tracks()
    names = []
    genres = []

    # for each artist and track add the artist as a key to the dictionary and add the track title to a list as the value
    for index, row in sample.iterrows():
        artist = row['artist_name']
        genre = row['genre_label']
        
        # Check if the artist is not in the dictionary
        if artist not in names:
            # Add the artist as a key
            names.append(artist)
            genres.append(genre)
    print(len(names))
    names = process_artist_names(names, True, True, False)
    processed_X = vectorise(names, "tfidf")

    descText = "TFIDF_lower_nostop_nopunc"
    # performs evaluating tests
    run_mega(genres, processed_X, descText)
