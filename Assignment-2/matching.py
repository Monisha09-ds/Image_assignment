import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def match_sift_features(desc1, desc2, ratio_test=0.75):
    """
    Given two sets of SIFT descriptors, returns a list of indices of matching pairs (i, j).
    Uses Lowe's ratio test for better matching quality.
    
    Args:
        desc1: numpy array of shape (N1, 128)
        desc2: numpy array of shape (N2, 128)
        ratio_test: threshold for Lowe's ratio test
        
    Returns:
        matches: list of tuples (idx1, idx2)
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
        
    # Calculate Euclidean distances between all pairs of descriptors
    # distances[i, j] is the distance between desc1[i] and desc2[j]
    distances = cdist(desc1, desc2, 'euclidean')
    
    matches = []
    for i in range(distances.shape[0]):
        # Get sorted indices for distance from desc1[i] to all desc2
        sorted_indices = np.argsort(distances[i, :])
        
        # Lowe's ratio test
        if len(sorted_indices) >= 2:
            best_dist = distances[i, sorted_indices[0]]
            second_best_dist = distances[i, sorted_indices[1]]
            
            if best_dist < ratio_test * second_best_dist:
                matches.append((i, sorted_indices[0]))
        elif len(sorted_indices) == 1:
            matches.append((i, sorted_indices[0]))
            
    return matches

def plot_matches(img1, img2, kp1, kp2, matches):
    """
    Combines two images side-by-side and plots matches between keypoints.
    
    Args:
        img1, img2: images to plot
        kp1, kp2: keypoint coordinates (N, 2)
        matches: list of tuples (idx1, idx2)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create combined image
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:w1+w2] = img2
    
    plt.figure(figsize=(15, 8))
    plt.imshow(combined_img)
    plt.axis('off')
    
    # Plot keypoints and matches
    for idx1, idx2 in matches:
        coord1 = kp1[idx1] # (col, row) or (x, y)? Usually kp are (y, x) or (x, y)
        coord2 = kp2[idx2]
        
        # Draw line between keypoints
        # Note: keypoint coordinates usually come as (row, col) or (col, row)
        # Assuming (x, y) / (col, row) for plot
        plt.plot(coord1[0], coord1[1], 'ro', markersize=3)
        plt.plot(coord2[0] + w1, coord2[1], 'ro', markersize=3)
        plt.plot([coord1[0], coord2[0] + w1], [coord1[1], coord2[1]], 'g-', linewidth=0.5)
        
    plt.title(f"SIFT Matches: {len(matches)} pairs")
    plt.show()
