import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================================================
# Leaf Mask Extraction (same logic as notebook)

def get_leaf_mask(rgb_img, dilate_px=1):
    """
    Generate a binary mask for leaf regions using HSV color thresholding.
    This follows the same approach used in the original notebook.
    """

    # Convert RGB → HSV
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    # Green color range (as used in the notebook)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Optional dilation (same idea as notebook visualization)
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# =========================================================
# Leaf-only image (background = black)
# Used in Stage-2 training & evaluation

def leaf_only_black(rgb_img):
    """
    Keep leaf pixels only, set background to black.
    Resize to (224,224) and apply EfficientNet preprocessing.
    """

    mask = get_leaf_mask(rgb_img, dilate_px=1)

    leaf_only = np.where(mask[..., None] > 0, rgb_img, 0)

    leaf_only = cv2.resize(
        leaf_only, (224, 224), interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    return preprocess_input(leaf_only)


# =========================================================
# Background-only image (leaf removed)
# Used for background-bias evaluation

def background_only(rgb_img):
    """
    Remove leaf pixels and keep background only.
    Resize to (224,224) and apply EfficientNet preprocessing.
    """

    mask = get_leaf_mask(rgb_img, dilate_px=1)

    bg_only = np.where(mask[..., None] > 0, 0, rgb_img)

    bg_only = cv2.resize(
        bg_only, (224, 224), interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    return preprocess_input(bg_only)
