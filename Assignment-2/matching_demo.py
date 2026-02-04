import cv2
import numpy as np
import matplotlib.pyplot as plt
from matching import match_sift_features, plot_matches
import os

def main():
    # Use images from previous assignment if available, else create dummy
    img1_path = "../Assignment -1/test_image.png"
    if not os.path.exists(img1_path):
        # Create dummy image
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img1, (20, 20), (80, 80), (255, 0, 0), -1)
        cv2.circle(img1, (50, 50), 10, (0, 255, 0), -1)
    else:
        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
    # Create img2 by rotating or shifting img1
    rows, cols = img1.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1) # Rotate 10 degrees
    img2 = cv2.warpAffine(img1, M, (cols, rows))
    
    # Detect SIFT features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Convert keypoints to numpy coordinates (x, y)
    kp1_pts = np.array([kp.pt for kp in kp1])
    kp2_pts = np.array([kp.pt for kp in kp2])
    
    print(f"Detected {len(kp1)} keypoints in Image 1")
    print(f"Detected {len(kp2)} keypoints in Image 2")
    
    # Match features
    matches = match_sift_features(des1, des2)
    print(f"Found {len(matches)} matches")
    
    # Plot
    plot_matches(img1, img2, kp1_pts, kp2_pts, matches)

if __name__ == "__main__":
    main()
