"""
Purpose: Evaluates normalized track duration as a single feature for genre classification,
         using StandardScaler for feature normalization before applying ML models.

Key Functions:
- main(): Processes and normalizes track duration data, then evaluates classification
          performance using multiple models (Naive Bayes, KNN)

Notes:
- Extends duration.py by adding feature normalization
- Uses StandardScaler for feature normalization
- Implements multiple classification algorithms for comparison
- Part of the single-feature analysis pipeline
- Helps understand if normalized duration values improve classification accuracy
"""

from sklearn.preprocessing import StandardScaler
from initialPreprocessing import gen_Train_and_Test, top_tracks
import pandas as pd
from svm import svm
from knn import knn
from nb import nb

if __name__ == "__main__":
    sample = top_tracks()
    X_train, X_test, y_train, y_test = gen_Train_and_Test(sample, 'track_duration', 0)
    
    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit the scaler to the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n\nTesting for Single Feature - duration")
    nb(X_train_scaled, X_test_scaled, y_train, y_test)
    knn(X_train_scaled, X_test_scaled, y_train, y_test)