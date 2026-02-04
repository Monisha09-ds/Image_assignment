# Assignment 2 Summary

## Quantitative Results (Placeholders)
*Note: These results depend on the completion of the dataset download and feature extraction.*

- **Number of features extracted using SIFT**: ~[Average per image]
- **Number of features extracted using HOG**: ~[Fixed by block size]
- **Number of correct matches found (Demo)**: [Variable]
- **SIFT SVM Accuracy**: [TBD]%
- **HOG SVM Accuracy**: [TBD]%

## Discussion Questions

### 1. Keypoint matching using HOG features
Describe a process for performing keypoint matching using HOG features. The challenge here is that HOG features are typically generated for the entire image.

**Response:**
To perform keypoint matching with HOG, we must transition from a global descriptor to a local one. The process involves:
1.  **Keypoint Detection**: Use a detector (e.g., Shi-Tomasi or a grid) to identify points of interest.
2.  **Local Neighborhood**: Define a small window (e.g., 16x16 pixels) around each detected keypoint.
3.  **Local HOG Extraction**: Compute the HOG descriptor for only that local window rather than the whole image.
4.  **Matching**: Compare these local descriptors between two images using a distance metric (like Euclidean distance) and a matcher (like Nearest Neighbor with Ratio Test).

### 2. Interpretation of results
Why do you think one feature set performed better than the other? Consider the efficiency of the feature extraction process and the quality of the features themselves.

**Response:**
- **SIFT**: Generally performs better for point-to-point matching and image stitching because of its invariance to scale and rotation. However, it can be computationally expensive to extract and might find very few points on low-resolution images like CIFAR-10 (32x32).
- **HOG**: Very efficient and excellent at capturing shape/gradient structures. For classification tasks on CIFAR-10, HOG often performs surprisingly well because it captures the overall "silhouette" of the object, which is sufficient for 32x32 images where fine-grained local details (SIFT) might be absent or noisy. 
- **Efficiency**: HOG is typically faster as it uses a fixed grid and simple gradient binning, whereas SIFT involves scale-space extrema detection which is more complex.
