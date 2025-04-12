"""
Purpose: Evaluates track favorite counts as a single feature for genre classification,
         analyzing if user engagement metrics correlate with music genres.
         
Key Functions:
- main(): Processes track favorite count data and evaluates classification performance
          using multiple models (SGD, KNN, Naive Bayes)
"""

from initialPreprocessing import gen_Train_and_Test, top_tracks
import pandas as pd
from svm import svm
from knn import knn
from nb import nb
from sgd import sgd

if __name__ == "__main__":                                                       
    sample = top_tracks()
    X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'track_favorites', 0)

    print("\n\nTesting for Single Feature - track favourites")
    # nb(X_train, X_test, y_train, y_test)
    # knn(X_train, X_test, y_train, y_test)
    sgd(X_train, X_test, y_train, y_test)
    