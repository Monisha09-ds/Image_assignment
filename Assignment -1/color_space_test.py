import sys
import os
import numpy as np
from PIL import Image
from color_conversions import rgb_to_hsv, hsv_to_rgb

def main():
    if len(sys.argv) < 5:
        print("Usage: python color_space_test.py <filename> <hue_mod> <sat_mod> <val_mod>")
        sys.exit(1)
        
    filename = sys.argv[1]
    try:
        hue_mod = float(sys.argv[2])
        sat_mod = float(sys.argv[3])
        val_mod = float(sys.argv[4])
    except ValueError:
        print("Error: Hue, Saturation, and Value modifications must be numbers.")
        sys.exit(1)
        
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
        
    # Validation
    if not (0 <= sat_mod <= 1) or not (0 <= val_mod <= 1):
        print("Warning: Saturation and Value modifications must be within range [0, 1].")
        print("Exiting program.")
        sys.exit(1)
        
    # Clamp Hue
    hue_mod = hue_mod % 360
    
    # Load Image
    try:
        img = Image.open(filename).convert('RGB')
        img_np = np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
        
    # Convert to HSV
    hsv = rgb_to_hsv(img_np)
    
    # Apply modifications (clamping results to valid ranges)
    hsv[:,:,0] = (hsv[:,:,0] + hue_mod) % 360
    hsv[:,:,1] = np.clip(hsv[:,:,1] + sat_mod, 0, 1)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + val_mod, 0, 1)
    
    # Convert back to RGB
    rgb = hsv_to_rgb(hsv)
    
    # Save modified image
    res_img = Image.fromarray((rgb * 255).astype(np.uint8))
    base, ext = os.path.splitext(filename)
    out_filename = f"{base}_modified{ext}"
    res_img.save(out_filename)
    
    print(f"Modified image saved as: {out_filename}")

if __name__ == "__main__":
    main()
