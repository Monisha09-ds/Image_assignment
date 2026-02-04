import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

def main():
    if not os.path.exists("sift_features.npz"):
        print("Error: sift_features.npz not found. Run feature_extraction.py first.")
        return
        
    data = np.load("sift_features.npz", allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    print("Training SVM on SIFT features...")
    clf = SVC()
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"SIFT SVM Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
