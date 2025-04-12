"""
Purpose: Implements Stochastic Gradient Descent classification with configurable loss functions
         (hinge loss for SVM-like behavior and modified huber loss for probability estimates).

Key Functions:
- sgd(X_train, X_test, y_train, y_test, desc=""): 
    Main implementation with full metrics reporting.
    Returns accuracy and predictions.

- sgd_(X_train, X_test, y_train, y_test, hard=True): 
    Configurable version with loss function selection.
    Returns classifier and predictions.
    hard=True uses hinge loss, False uses modified huber loss.

Notes:
- Uses sklearn's SGDClassifier with fixed random state for reproducibility
- Provides detailed classification metrics for model evaluation
"""

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# The sgd function provides full metrics reporting for standalone model evaluation
def sgd(X_train, X_test, y_train, y_test, desc=""):
    print("\n\n\nSGD"+desc+":\n")
    print(str(X_train.shape[1]) + " features")
    
    sgd_clf = SGDClassifier(loss='hinge', random_state=42)
    sgd_clf.fit(X_train, y_train)  # Train the classifier
    
    y_pred = sgd_clf.predict(X_test)  # Predict using the trained classifier
    
    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)  # Generate classification report
    
    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    return accuracy,y_pred

# The sgd_ function is a configurable version for flexible loss selection and ensemble integration 
def sgd_(X_train, X_test, y_train, y_test, hard=True):
    print("\n\n\nSGD\n")
    print(str(X_train.shape[1]) + " features")
    
    if hard:
        sgd_clf = SGDClassifier(loss='hinge', random_state=42)
    else:
        sgd_clf = SGDClassifier(loss='modified_huber', random_state=42)

    sgd_clf.fit(X_train, y_train)  # Train the classifier
    
    y_pred = sgd_clf.predict(X_test)  # Predict using the trained classifier

    return sgd_clf,y_pred