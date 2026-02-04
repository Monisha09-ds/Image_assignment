import numpy as np
import cv2
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def extract_sift_descriptors(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    
    for img in images:
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        kp, des = sift.detectAndCompute(gray, None)
        if des is not None:
            descriptors_list.append(des)
        else:
            # If no SIFT features found, append an empty array
            descriptors_list.append(np.array([]).reshape(0, 128))
            
    return descriptors_list

def extract_hog_descriptors(images):
    descriptors_list = []
    for img in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Extract HOG from 8x8 blocks as local descriptors for BoVW
        fd = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False, feature_vector=False)
        
        # fd shape is (n_blocks_row, n_blocks_col, 2, 2, 9)
        # Reshape to (M, 36) where M is number of blocks
        m, n, _, _, _ = fd.shape
        descriptors = fd.reshape(-1, 36)
        descriptors_list.append(descriptors)
        
    return descriptors_list

def create_visual_vocabulary(descriptors_list, vocab_size=100):
    # Flatten all descriptors from all images into one large array
    all_descriptors = np.vstack([d for d in descriptors_list if d.shape[0] > 0])
    
    # Use a subset if there are too many descriptors (for performance)
    if all_descriptors.shape[0] > 10000:
        indices = np.random.choice(all_descriptors.shape[0], 10000, replace=False)
        all_descriptors = all_descriptors[indices]
        
    kmeans = KMeans(n_clusters=vocab_size, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)
    return kmeans

def quantize_to_histogram(descriptors_list, kmeans):
    vocab_size = kmeans.n_clusters
    histograms = []
    
    for des in descriptors_list:
        hist = np.zeros(vocab_size)
        if des.shape[0] > 0:
            # Predict labels for each descriptor
            labels = kmeans.predict(des)
            # Build histogram
            for label in labels:
                hist[label] += 1
        
        # L2 Normalize histogram
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
            
        histograms.append(hist)
        
    return np.array(histograms)

def process_and_save(X_train, y_train, X_test, y_test, feature_type='sift'):
    print(f"Processing {feature_type} features...")
    
    if feature_type == 'sift':
        train_descriptors = extract_sift_descriptors(X_train)
        test_descriptors = extract_sift_descriptors(X_test)
        vocab_size = 100 
    else: # hog
        train_descriptors = extract_hog_descriptors(X_train)
        test_descriptors = extract_hog_descriptors(X_test)
        vocab_size = 50 
        
    print("Creating visual vocabulary...")
    kmeans = create_visual_vocabulary(train_descriptors, vocab_size=vocab_size)
    
    print("Quantizing into histograms...")
    X_train_bovw = quantize_to_histogram(train_descriptors, kmeans)
    X_test_bovw = quantize_to_histogram(test_descriptors, kmeans)
    
    save_path = f"{feature_type}_features.npz"
    np.savez(save_path, X_train=X_train_bovw, y_train=y_train, 
             X_test=X_test_bovw, y_test=y_test)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    if not os.path.exists("cifar10.npz"):
        print("Error: cifar10.npz not found. Run load_and_split.py first.")
    else:
        # Load the pre-split data
        data = np.load("cifar10.npz", allow_pickle=True)
        
        X_train_raw = data["X_train"]
        y_train = data["y_train"]
        X_test_raw = data["X_test"]
        y_test = data["y_test"]
        
        # Reshape to (N, 32, 32, 3)
        def to_images(X):
            # If X is a DataFrame (from openml), convert to numpy
            if hasattr(X, "to_numpy"):
                X = X.to_numpy()
            
            if len(X.shape) == 2:
                N = X.shape[0]
                # CIFAR-10 is often RRR...GGG...BBB...
                imgs = X.reshape(N, 3, 32, 32).transpose(0, 2, 3, 1)
                return imgs.astype(np.uint8)
            return X

        X_train = to_images(X_train_raw)
        X_test = to_images(X_test_raw)
        
        print(f"X_train shape: {X_train.shape}")
        
        # Process both SIFT and HOG
        process_and_save(X_train, y_train, X_test, y_test, feature_type='sift')
        process_and_save(X_train, y_train, X_test, y_test, feature_type='hog')