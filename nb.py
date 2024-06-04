from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


def nb(X_train, X_test, y_train, y_test, desc=""):
    print("\n\n\nNB"+desc+":\n")
    print(str(X_train.shape[1]) + " features")
    # Create a Gaussian Naive Bayes classifier
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train, y_train) # train data
    y_pred = naive_bayes_classifier.predict(X_test) # predict data
    
    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0) # Generate report of the values
    
    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    return accuracy, y_pred

def nb_(X_train, X_test, y_train, y_test, desc=""):
    print("\n\n\nNB"+desc+":\n")
    print(str(X_train.shape[1]) + " features")
    # Create a Gaussian Naive Bayes classifier
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train, y_train) # train data
    y_pred = naive_bayes_classifier.predict(X_test) # predict data
    

    return naive_bayes_classifier, y_pred