import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

def main():
    if not os.path.exists("hog_features.npz"):
        print("Error: hog_features.npz not found. Run feature_extraction.py first.")
        return
        
    data = np.load("hog_features.npz", allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    print("Training SVM on HOG features...")
    # Using linear kernel for speed, but SVC default is rbf
    # Prompt says "You should use the default parameters for the SVM"
    clf = SVC()
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"HOG SVM Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
