import numpy as np

def rgb_to_hsv(rgb):
    """
    Converts an RGB image represented as a numpy array into HSV format.
    Implementation is vectorized for performance.
    
    Args:
        rgb: numpy array of shape (H, W, 3) with values in [0, 255] or [0, 1]
        
    Returns:
        hsv: numpy array of shape (H, W, 3) 
             H in [0, 360], S in [0, 1], V in [0, 1]
    """
    # Ensure float precision and range [0, 1]
    rgb = rgb.astype(np.float32) / 255.0 if rgb.max() > 1.0 else rgb.astype(np.float32)
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    
    v = np.amax(rgb, axis=2)
    m = np.amin(rgb, axis=2)
    delta = v - m
    
    # Saturation
    s = np.zeros_like(v)
    mask = v > 0
    s[mask] = delta[mask] / v[mask]
    
    # Hue
    h = np.zeros_like(v)
    
    # R is max
    mask_r = (v == r) & (delta > 0)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    
    # G is max
    mask_g = (v == g) & (delta > 0)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    
    # B is max
    mask_b = (v == b) & (delta > 0)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    
    # Hue in [0, 360)
    h = h % 360
    
    return np.stack([h, s, v], axis=2)

def hsv_to_rgb(hsv):
    """
    Converts an HSV image represented as a numpy array into RGB format.
    Implementation is vectorized for performance.
    
    Args:
        hsv: numpy array of shape (H, W, 3)
             H in [0, 360], S in [0, 1], V in [0, 1]
             
    Returns:
        rgb: numpy array of shape (H, W, 3) with values in [0, 1]
    """
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    c = v * s
    x = c * (1 - np.abs((h / 60.0) % 2 - 1))
    m = v - c
    
    # Piecewise conditions for H
    cond0 = (h >= 0) & (h < 60)
    cond1 = (h >= 60) & (h < 120)
    cond2 = (h >= 120) & (h < 180)
    cond3 = (h >= 180) & (h < 240)
    cond4 = (h >= 240) & (h < 300)
    cond5 = (h >= 300) & (h < 360)
    
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    
    # R bits
    r = np.select([cond0, cond1, cond2, cond3, cond4, cond5], 
                  [c, x, 0, 0, x, c], default=0)
    # G bits
    g = np.select([cond0, cond1, cond2, cond3, cond4, cond5],
                  [x, c, c, x, 0, 0], default=0)
    # B bits
    b = np.select([cond0, cond1, cond2, cond3, cond4, cond5],
                  [0, 0, x, c, c, x], default=0)
    
    rgb = np.stack([r + m, g + m, b + m], axis=2)
    return np.clip(rgb, 0, 1)
