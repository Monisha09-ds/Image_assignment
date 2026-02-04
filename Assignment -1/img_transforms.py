import numpy as np
import random
from color_conversions import rgb_to_hsv, hsv_to_rgb

def random_crop(img, size):
    """
    Generates a random square crop of an image.
    
    Args:
        img: numpy array of shape (H, W, 3)
        size: integer reflecting the size of the square crop (s x s)
        
    Returns:
        cropped_img: numpy array of shape (size, size, 3)
    """
    h, w = img.shape[:2]
    
    if size <= 0 or size > min(h, w):
        raise ValueError(f"Crop size {size} is not feasible for image of size {w}x{h}")
    
    # Randomly pick top-left corner
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)
    
    return img[top:top+size, left:left+size, :]

def extract_patch(img, num_patches):
    """
    Returns n^2 non-overlapping patches from a square image.
    
    Args:
        img: numpy array of shape (H, H, 3) (assumed square)
        num_patches: integer n, creating n^2 patches
        
    Returns:
        patches: list of numpy arrays, each of shape (H/n, H/n, 3)
    """
    h, w = img.shape[:2]
    if h != w:
        # If not square, we take the minimum dimension to ensure simple patching
        # but the prompt says "You may assume that the input image is square."
        pass
    
    patch_size = h // num_patches
    patches = []
    
    for i in range(num_patches):
        for j in range(num_patches):
            top = i * patch_size
            left = j * patch_size
            patch = img[top:top+patch_size, left:left+patch_size, :]
            patches.append(patch)
            
    return patches

def resize_img(img, factor):
    """
    Resizes an image using nearest neighbor interpolation.
    
    Args:
        img: numpy array
        factor: integer scale factor (e.g., 2 means half size, or twice size?)
                Usually "scale factor" in pyramids implies powers of 2 (1/2, 1/4, etc?)
                The prompt says "resized copies of the original image... in powers of 2".
                And Figure 3 shows "resized with different scale factors".
                If factor is an integer, maybe it means target size is img.shape / factor?
                Let's assume factor is a multiplier for coordinates (e.g. factor 2 means 2x larger).
                Wait, pyramids say "128, 64, 32" for a "256" image. That's factor 2, 4, 8 reduction.
                Let's implement it such that it can handle any float factor for flexibility,
                but specifically nearest neighbor.
    """
    h, w = img.shape[:2]
    # If factor is 2, and we want 1/2 size: new_h = h // factor
    # Let's interpret factor as "downscale factor" if > 1 for pyramids, 
    # but the tool needs to be general.
    # To be safe, if factor is > 0, new_w = int(w * factor), new_h = int(h * factor)
    # Actually, the pyramid prompt says "resized copies... in powers of 2... 128, 64, 32 for 256".
    # So if scale factor appended is 2x, 4x, 8x, then the image is small.
    # So factor=2 means new_size = old_size / 2.
    
    new_h, new_w = int(h / factor), int(w / factor)
    
    # Vectorized Nearest Neighbor
    row_indices = (np.arange(new_h) * factor).astype(int)
    col_indices = (np.arange(new_w) * factor).astype(int)
    
    # Clip indices to avoid out of bounds
    row_indices = np.clip(row_indices, 0, h - 1)
    col_indices = np.clip(col_indices, 0, w - 1)
    
    return img[np.ix_(row_indices, col_indices)]

def color_jitter(img, hue, saturation, value):
    """
    Randomly perturbs the HSV values of an image.
    
    Args:
        img: numpy array (RGB)
        hue: float (max random perturbation for H)
        saturation: float (max random perturbation for S)
        value: float (max random perturbation for V)
    """
    # Convert to HSV
    hsv = rgb_to_hsv(img)
    
    # Generate random perturbations in range [-val, val]
    h_rand = random.uniform(-hue, hue)
    s_rand = random.uniform(-saturation, saturation)
    v_rand = random.uniform(-value, value)
    
    # Apply and clamp
    hsv[:,:,0] = (hsv[:,:,0] + h_rand) % 360
    hsv[:,:,1] = np.clip(hsv[:,:,1] + s_rand, 0, 1)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + v_rand, 0, 1)
    
    # Convert back to RGB
    rgb = hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8)
