# =============================================================
#  utils.py  —  CV Workshop: Image Utilities
#  Part 1 of 3
# =============================================================
#
#  Implement every function below using OpenCV and NumPy.
#  Your only guide is the docstring for each function.
#
#  Self-test when you think you're done:
#      python utils.py
#
#  Reference: https://docs.opencv.org/4.x/
# =============================================================

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and return it as a NumPy array.

    OpenCV loads images in BGR channel order, not RGB.
    Raise FileNotFoundError with a descriptive message if the
    file cannot be read.

    Args:
        path: Path to a .jpg or .png file.

    Returns:
        ndarray of shape (H, W, 3), dtype uint8.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert a BGR color image to a single-channel grayscale image.

    Args:
        img: ndarray of shape (H, W, 3).

    Returns:
        ndarray of shape (H, W), dtype uint8.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise NotImplementedError


def blur_image(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to a grayscale image to suppress noise.

    The kernel_size must be a positive odd integer.

    Args:
        gray:        Single-channel image, shape (H, W).
        kernel_size: Blur kernel size. Default 5.

    Returns:
        ndarray of same shape as gray, dtype uint8.
    """
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    raise NotImplementedError


def detect_edges(blurred: np.ndarray,
                 low_threshold: int = 50,
                 high_threshold: int = 150) -> np.ndarray:
    """
    Run Canny edge detection on a blurred grayscale image.

    Args:
        blurred:        Blurred single-channel image, shape (H, W).
        low_threshold:  Lower hysteresis threshold.
        high_threshold: Upper hysteresis threshold.

    Returns:
        Binary ndarray of same shape, values 0 or 255.
    """
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges
    raise NotImplementedError


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Run the full preprocessing pipeline on a color image.

    Chain, in order: grayscale → blur → edge detection.
    Call the three functions above — do not re-implement the logic.

    Args:
        img: BGR color image, shape (H, W, 3).

    Returns:
        Binary edge map, shape (H, W), values 0 or 255.
    """
    gray = to_grayscale(img)
    blurred = blur_image(gray)
    edges = detect_edges(blurred)
    return edges
    raise NotImplementedError


def find_subject_contour(edges: np.ndarray, min_area: int = 5000):
    """
    Extract all external contours from the edge map, discard any
    whose area is below min_area, and return the largest remaining
    contour. Return None if no contour qualifies.

    Args:
        edges:    Binary edge map, shape (H, W).
        min_area: Minimum contour area in px². Default 5000.

    Returns:
        The largest qualifying contour (ndarray of points), or None.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Filter contours by pixel area 
    large_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # 3. Handle cases where no subjects meet the size requirement
    if not large_contours:
        return None

    # 4. Use max() with cv2.contourArea as the key to find the biggest subject [cite: 25]
    largest_contour = max(large_contours, key=cv2.contourArea)
    
    return largest_contour


def crop_roi(img: np.ndarray, contour) -> tuple:
    """
    Crop the bounding rectangle of a contour from the original
    color image. Return both the crop and the box coordinates.

    The crop must come from the COLOR image, not the edge map —
    the classifier needs texture and color information.

    Args:
        img:     BGR color image, shape (H, W, 3).
        contour: A contour returned by find_subject_contour.

    Returns:
        roi: Cropped color image, shape (h, w, 3).
        box: Tuple (x, y, w, h) in pixels.
    """
    x, y, w, h = cv2.boundingRect(contour)
    roi = img[y:y+h, x:x+w]
    return roi, (x, y, w, h)
    raise NotImplementedError


# =============================================================
#  SELF-TEST — python utils.py
#  All checks must pass before you move to model.py.
# =============================================================
if __name__ == "__main__":
    import sys, os

    test_path = "images/dog.jpg"
    if not os.path.exists(test_path):
        print(f"[!] No test image at '{test_path}'.")
        print("    Make sure you cloned the full repo (images/ folder included).")
        sys.exit(1)

    print("Testing utils.py ...\n")

    try:
        img = load_image(test_path)
        assert img is not None and img.ndim == 3 and img.shape[2] == 3
        print(f"  [✓] load_image            → shape {img.shape}, dtype {img.dtype}")
    except Exception as e:
        print(f"  [✗] load_image            → {e}"); sys.exit(1)

    try:
        gray = to_grayscale(img)
        assert gray.ndim == 2, f"Expected 2D array, got {gray.ndim}D"
        print(f"  [✓] to_grayscale          → shape {gray.shape}")
    except Exception as e:
        print(f"  [✗] to_grayscale          → {e}"); sys.exit(1)

    try:
        blurred = blur_image(gray)
        assert blurred.shape == gray.shape
        print(f"  [✓] blur_image            → shape {blurred.shape}")
    except Exception as e:
        print(f"  [✗] blur_image            → {e}"); sys.exit(1)

    try:
        edges = detect_edges(blurred)
        unique = set(np.unique(edges))
        assert unique.issubset({0, 255}), f"Not binary — found values: {unique}"
        print(f"  [✓] detect_edges          → shape {edges.shape}, binary ✓")
    except Exception as e:
        print(f"  [✗] detect_edges          → {e}"); sys.exit(1)

    try:
        edges2 = preprocess(img)
        assert edges2.shape == (img.shape[0], img.shape[1])
        print(f"  [✓] preprocess            → shape {edges2.shape}")
    except Exception as e:
        print(f"  [✗] preprocess            → {e}"); sys.exit(1)

    try:
        contour = find_subject_contour(edges2)
        if contour is not None:
            print(f"  [✓] find_subject_contour  → area {cv2.contourArea(contour):.0f} px²")
        else:
            print("  [~] find_subject_contour  → None (try a clearer photo)")
    except Exception as e:
        print(f"  [✗] find_subject_contour  → {e}"); sys.exit(1)

    if contour is not None:
        try:
            roi, box = crop_roi(img, contour)
            assert roi.ndim == 3 and roi.shape[2] == 3
            assert len(box) == 4
            print(f"  [✓] crop_roi              → roi {roi.shape}, box {box}")
        except Exception as e:
            print(f"  [✗] crop_roi              → {e}"); sys.exit(1)

    print("\n✓ All utils.py tests passed — move on to model.py\n")