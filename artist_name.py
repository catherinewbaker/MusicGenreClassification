from initialPreprocessing import gen_Train_and_Test, top_tracks
import pandas as pd
from svm import svm
from knn import knn
from nb import nb
from sgd import sgd

# will need to install nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string


def process_artist_names(df, lowercase=False,use_stop_words=False, remove_punctuation=False):
    stemmer = PorterStemmer()

    if use_stop_words:
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set()

    processed_names = []

    for name in df['artist_name']:
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
    # # knn(X_train, X_test, y_train, y_test,descText,99)
    # # knn(X_train, X_test, y_train, y_test,descText,199)
    # nb(X_train, X_test, y_train, y_test,descText)
    sgd(X_train, X_test, y_train, y_test,descText)


if __name__ == "__main__":                                                       
    sample = top_tracks()
    # sample = sample.sample(n=10000,random_state=42)

    # Only need these 2 lines the first time you run it
    # nltk.download('punkt')
    # nltk.download('stopwords')

    # keep stop words but remove punctuation
    sample_1 = process_artist_names(sample, True, False, True)
    # # keep stop words and punctuation
    # sample_4 = process_track_names(sample, True, False, False)
   

    sample_1_tfidf = vectorise(sample_1,"tfidf")
    descText=" TF-IDF_lower_nostop_nopunc"

    run_mega(sample,sample_1_tfidf,descText)
    
    sample_1_tfidf = vectorise(sample_1,"bow")
    descText=" bow_lower_nostop_nopunc"

    run_mega(sample,sample_1_tfidf,descText)