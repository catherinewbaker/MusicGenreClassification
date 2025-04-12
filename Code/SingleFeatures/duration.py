"""
Purpose: Evaluates track duration as a single feature for genre classification using
         multiple machine learning models to assess its predictive power.

Key Functions:
- main(): Processes track duration data and evaluates classification performance
          using multiple models (Naive Bayes, KNN, SGD)

Notes:
- Implements multiple classification algorithms for comparison
"""

from initialPreprocessing import gen_Train_and_Test, top_tracks, top_n_genre_tracks, top_echonest_tracks, top_tracks_final
import pandas as pd
from svm import svm
from knn import knn
from nb import nb
from sgd import sgd

if __name__ == "__main__":                                                       
    sample = top_tracks_final()
    # sample = top_echonest_tracks(True)

    X_train, X_test, y_train, y_test = gen_Train_and_Test(sample[0],'track_duration',0)

    print("\n\nTesting for Single Feature - duration")
    nb(X_train, X_test, y_train, y_test)
    knn(X_train, X_test, y_train, y_test)
    sgd(X_train, X_test, y_train, y_test)


