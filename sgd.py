from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


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