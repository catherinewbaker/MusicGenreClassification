"""
Purpose: Processes and vectorizes track names for music classification using various text preprocessing techniques
         and machine learning models (KNN, Naive Bayes, SGD)

Key Functions:
- process_track_names(): Preprocesses track titles with options for lowercase conversion, stopword removal, and punctuation handling
- vectorise(): Converts processed text to numerical features using either TF-IDF or Bag of Words
- run_mega(): Runs multiple machine learning models on the processed and vectorized data

Notes:
- Implements multiple text preprocessing combinations (with/without stopwords, punctuation)
- Uses NLTK for text processing and scikit-learn for vectorization
- Supports both TF-IDF and Bag of Words (BOW) vectorization approaches
"""

from initialPreprocessing import gen_Train_and_Test, top_tracks
import pandas as pd
from svm import svm
from knn import knn
from nb import nb
from sgd import sgd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string


def process_track_names(df, lowercase=False,use_stop_words=False, remove_punctuation=False):
    stemmer = PorterStemmer()

    if use_stop_words:
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set()

    processed_titles = []

    for title in df['track_title']:
        thistitle = title
        if lowercase:
            thistitle = thistitle.lower()
        
        tokens = word_tokenize(thistitle)
        
        if remove_punctuation:
            processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token not in string.punctuation]
        else:
            processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

        processed_title = ' '.join(processed_tokens)
        processed_titles.append(processed_title)

    return processed_titles

def vectorise(titles, method):
    if method=="tfidf":
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
        tfidf_matrix_dense = tfidf_matrix.toarray()
        return tfidf_matrix_dense
    elif method == "bow":
        count_vectorizer = CountVectorizer()
        bow_matrix = count_vectorizer.fit_transform(titles)
        bow_matrix_dense = bow_matrix.toarray()
        return bow_matrix_dense
    else:
        "Please specify a valid vectorisation approach"

def run_mega(sample, processed ,descText):
    X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0, processed)

    # knn(X_train, X_test, y_train, y_test,descText,1)
    # # knn(X_train, X_test, y_train, y_test,descText,3)
    # # knn(X_train, X_test, y_train, y_test,descText,5)
    # # knn(X_train, X_test, y_train, y_test,descText,7)
    # knn(X_train, X_test, y_train, y_test,descText,9)
    # knn(X_train, X_test, y_train, y_test,descText,21)
    # knn(X_train, X_test, y_train, y_test,descText,99)
    # knn(X_train, X_test, y_train, y_test,descText,199)
    # nb(X_train, X_test, y_train, y_test,descText)
    sgd(X_train, X_test, y_train, y_test)


if __name__ == "__main__":                                                       
    sample = top_tracks()
    # sample = sample.sample(n=10000,random_state=42)

    # Only need these 2 lines the first time you run it
    # nltk.download('punkt')
    # nltk.download('stopwords')

    sample_1 = process_track_names(sample, True, True, True)
    # keep punctuation
    sample_2 = process_track_names(sample, True, True, False)
    # keep stop words
    sample_3 = process_track_names(sample, True, False, True)
    # keep stop words and punctuation
    sample_4 = process_track_names(sample, True, False, False)
   

    # # SAMPLE 1
    # sample_1_tfidf = vectorise(sample_1,"tfidf")
    # descText=" TF-IDF_lower_stop_nopunc"

    # run_mega(sample,sample_1_tfidf,descText)
    

    # sample_1_bow = vectorise(sample_1,"bow")
    # descText=" BOW_lower_stop_nopunc"
    # run_mega(sample,sample_1_bow,descText)

    # # # SAMPLE 2
    # sample_2_tfidf = vectorise(sample_2,"tfidf")
    # descText=" TF-IDF_lower_stop_punc"

    # run_mega(sample,sample_2_tfidf,descText)
    

    # sample_2_bow = vectorise(sample_2,"bow")
    # descText=" BOW_lower_stop_punc"
    # run_mega(sample,sample_2_bow,descText)

    # # # SAMPLE 3
    # sample_3_tfidf = vectorise(sample_3,"tfidf")
    # descText=" TF-IDF_lower_nostop_nopunc"

    # run_mega(sample,sample_3_tfidf,descText)
    

    sample_3_bow = vectorise(sample_3,"bow")
    descText=" BOW_lower_nostop_nopunc"
    run_mega(sample,sample_3_bow,descText)

    # SAMPLE 4
    sample_4_tfidf = vectorise(sample_4,"tfidf")
    descText=" TF-IDF_lower_nostop_punc"

    run_mega(sample,sample_4_tfidf,descText)
    

    sample_4_bow = vectorise(sample_4,"bow")
    descText=" BOW_lower_nostop_punc"
    run_mega(sample,sample_4_bow,descText)
    
    
  


    # # print("\n\n\nSVM:\n")
    # # svm(X_train, X_test, y_train, y_test)
    