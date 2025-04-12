"""
Purpose: Implements K-Nearest Neighbors classification for music genre prediction, featuring both fixed-k 
         and automatic k-value optimization through grid search cross-validation.

Key Functions:
- knn(X_train, X_test, y_train, y_test, desc="", N=-1): 
    Main implementation with automatic k optimization. When N=-1 (default), performs grid search to find optimal k.
    Returns accuracy, predictions, and best k value.

- knn_(X_train, X_test, y_train, y_test, K): 
    Simplified version with fixed K value.
    Returns classifier and predictions without metrics.

Notes:
- Adaptive k-value range based on dataset size (sqrt of training samples)
- Performance optimization for high-dimensional data (>50 features)
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

import numpy as np

# The knn function implements KNN with automatic k optimization and full metrics reporting
def knn(X_train, X_test, y_train, y_test, desc="", N=-1):
    bestN = -1

    # Smaller nested CV - finds the best k value for high or low dimensional data
    if N<0:
        sqrt_sample_size = int(np.sqrt(X_train.shape[0]))

        if X_train.shape[1] > 50:
            print("CHECKING 10 K VALUES ONLY \n")
            k_values = np.linspace(1, sqrt_sample_size, 10, dtype=int)  # Check only 10 k values
        else:
            print("CHECKING ALL ODD K VALUES")
            k_values = list(range(1, sqrt_sample_size + 1, 2))  # Check all odd k values


        param_grid = {'n_neighbors': k_values}

        knn_classifier = KNeighborsClassifier()
    
        # Perform GridSearchCV
        grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)  # 5-fold cross-validation
        grid_search.fit(X_train, y_train)  # Fit the model with training data

        # Best hyperparameters
        best_params = grid_search.best_params_
        best_k = best_params['n_neighbors']

        # Train the model with the best k value
        best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
        best_knn_classifier.fit(X_train, y_train)  # Train data
        y_pred = best_knn_classifier.predict(X_test)  # Predict data

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)  # Generate report

        # Print the results
        print(f"Best k: {best_k}")
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)

        return accuracy, y_pred, best_k

        
    num_feats = N

    if bestN>0:
        num_feats = bestN


    print("\n\n\nKNN"+"("+str(num_feats)+")"+desc+":\n")
    print(str(X_train.shape[1]) + " features")

    # Create a K Nearest Neighbors Classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=num_feats)
    knn_classifier.fit(X_train, y_train) # train data
    y_pred = knn_classifier.predict(X_test) # predict data

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)  # Generate report of the values

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    return accuracy,y_pred, num_feats

# The knn_ function is a simplified version with fixed K for ensemble model integration
def knn_(X_train, X_test, y_train, y_test, K):

        
    print("\n\n\nKNN"+"("+str(K)+")")
    print(str(X_train.shape[1]) + " features")

    # Create a K Nearest Neighbors Classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=K)
    knn_classifier.fit(X_train, y_train) # train data
    y_pred = knn_classifier.predict(X_test) # predict data

    return knn_classifier,y_pred