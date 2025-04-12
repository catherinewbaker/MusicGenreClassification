"""
Purpose: Analyzes track creation dates as a feature for genre classification by
         converting dates to numerical values (days since earliest track) and
         applying various machine learning models.

Key Functions:
- gen_Train_and_Test(): Processes date features and generates train/test splits
                        by converting dates to days-since-first-track metric

Notes:
- Handles datetime conversion and normalization
- Implements data cleaning for null date values
- Supports optional dataset subsetting for experimentation
"""

from initialPreprocessing import top_tracks
from sklearn.model_selection import train_test_split
import pandas as pd
from svm import svm
from knn import knn
from nb import nb

def gen_Train_and_Test(data, feature, subset):
    if subset != 0:
        dataset = data.sample(n=subset, random_state=42)
    else:
        dataset = data

    # Ensure the 'track_date_created' column is a datetime object
    dataset[feature] = pd.to_datetime(dataset[feature])

    # Calculate the number of days since the first date in the dataset
    min_date = dataset[feature].min()
    dataset['days_since_first'] = (dataset[feature] - min_date).dt.days

    # Select the new feature 'days_since_first' for X
    X = dataset[['days_since_first']]
    y = dataset['genre_label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Remove any rows where 'days_since_first' is null from the training and testing sets
    train_mask = X_train['days_since_first'].notna()
    test_mask = X_test['days_since_first'].notna()

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    print("Training sample length:", len(X_train))
    print("Testing sample length:", len(X_test))

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    sample = top_tracks()
    X_train, X_test, y_train, y_test = gen_Train_and_Test(sample, 'track_date_created', 0)

    print("\n\nTesting for Single Feature - track_date_created")

    nb(X_train, X_test, y_train, y_test)

    knn(X_train, X_test, y_train, y_test)
