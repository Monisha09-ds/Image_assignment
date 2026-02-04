import numpy as np
from PIL import Image
import os
import img_transforms
import color_conversions

def main():
    # 1. Create a dummy test image if no image is provided, 
    # but for this demo, let's look for any .png or .jpg in the dir
    test_img_path = "test_image.png"
    if not os.path.exists(test_img_path):
        # Create a 256x256 color gradient image for testing
        h, w = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                img[i, j] = [i % 256, j % 256, (i + j) % 256]
        Image.fromarray(img).save(test_img_path)
        print(f"Created dummy test image: {test_img_path}")

    img = Image.open(test_img_path).convert('RGB')
    img_np = np.array(img)
    
    print("\n--- Testing Image Transformations ---")
    
    # Random Crop
    try:
        cropped = img_transforms.random_crop(img_np, 100)
        Image.fromarray(cropped).save("demo_crop.png")
        print("Random Crop (100x100) saved to demo_crop.png")
    except Exception as e:
        print(f"Random Crop failed: {e}")
        
    # Patch Extraction
    try:
        patches = img_transforms.extract_patch(img_np, 4) # 4x4 = 16 patches
        print(f"Extracted {len(patches)} patches.")
        # Save first patch as example
        Image.fromarray(patches[0]).save("demo_patch_0.png")
        print("First patch saved to demo_patch_0.png")
    except Exception as e:
        print(f"Patch Extraction failed: {e}")
        
    # Resizing (Nearest Neighbor)
    try:
        resized = img_transforms.resize_img(img_np, 2) # 0.5x scale
        Image.fromarray(resized.astype(np.uint8)).save("demo_resized_0.5x.png")
        print("Resized (0.5x) saved to demo_resized_0.5x.png")
    except Exception as e:
        print(f"Resizing failed: {e}")
        
    # Color Jitter
    try:
        jittered = img_transforms.color_jitter(img_np, 30, 0.2, 0.2)
        Image.fromarray(jittered).save("demo_jitter.png")
        print("Color Jitter saved to demo_jitter.png")
    except Exception as e:
        print(f"Color Jitter failed: {e}")
        
    print("\n--- Testing Color Conversions ---")
    try:
        hsv = color_conversions.rgb_to_hsv(img_np)
        rgb_back = color_conversions.hsv_to_rgb(hsv)
        diff = np.abs(img_np.astype(float)/255.0 - rgb_back)
        print(f"Color Conversion Max Difference: {np.max(diff)}")
    except Exception as e:
        print(f"Color Conversion failed: {e}")

    print("\nDemo complete. Check the generated demo_*.png files.")

if __name__ == "__main__":
    main()
