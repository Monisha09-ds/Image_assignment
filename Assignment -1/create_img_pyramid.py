import sys
import os
import numpy as np
from PIL import Image
from img_transforms import resize_img

def create_pyramid(img_np, height):
    """
    Creates an image pyramid of resized copies.
    
    Args:
        img_np: numpy array
        height: total height of the pyramid (including original)
                e.g., height 4 on 256x256 creates 128x128, 64x64, 32x32
                (original is level 1, then height-1 levels are created)
    """
    pyramid = []
    for i in range(1, height):
        factor = 2**i
        resized = resize_img(img_np, factor)
        pyramid.append((factor, resized))
    return pyramid

def main():
    if len(sys.argv) < 3:
        print("Usage: python create_img_pyramid.py <filename> <height>")
        sys.exit(1)
        
    filename = sys.argv[1]
    try:
        height = int(sys.argv[2])
    except ValueError:
        print("Error: Height must be an integer.")
        sys.exit(1)
        
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
        
    # Load Image
    try:
        img = Image.open(filename).convert('RGB')
        img_np = np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
        
    # Create Pyramid
    pyramid = create_pyramid(img_np, height)
    
    # Save copies
    base, ext = os.path.splitext(filename)
    for factor, resized in pyramid:
        out_filename = f"{base}_{factor}x{ext}"
        res_img = Image.fromarray(resized.astype(np.uint8))
        res_img.save(out_filename)
        print(f"Saved: {out_filename}")

if __name__ == "__main__":
    main()
