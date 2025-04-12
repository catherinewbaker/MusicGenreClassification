"""
Purpose: Implements Support Vector Machine classification using sklearn's SVC with linear kernel
         for multi-class music genre classification.

Key Functions:
- svm(X_train, X_test, y_train, y_test, desc=""): 
    Implements linear SVM classification with comprehensive metrics reporting.
    Returns predictions and prints accuracy and classification report.

Notes:
- Handles multi-class classification with built-in one-vs-rest approach
- Fixed random state (42) for reproducibility
- Uses LabelEncoder for consistent class label handling
"""

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings

def svm(X_train, X_test, y_train, y_test, desc=""):
    print("\n\n\nSVM"+desc+":\n")
    print(str(X_train.shape[1]) + " features")
    
    warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    # Create an SVM Classifier
    svm_model = SVC(kernel='linear',random_state=42)
    svm_model.fit(X_train,y_train) # train data
    y_pred = svm_model.predict(X_test) # predict data

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test,y_pred)

    # Fixing class labels for the report generation
    unique_classes = sorted(set(y_test) | set(y_pred))
    label_encoder = LabelEncoder()
    report = classification_report(y_test, y_pred, labels=unique_classes, target_names=label_encoder.classes_, zero_division=0) # Generate report of the values

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    return y_pred