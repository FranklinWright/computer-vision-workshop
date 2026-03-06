# =============================================================
#  model.py  —  CV Workshop: Model Loading & Inference
#  Part 2 of 3
#
#  Model: GoogLeNet trained on ImageNet (Caffe)
#  Files needed in the repo root:
#      bvlc_googlenet.caffemodel   (~50 MB, weights)
#      deploy.prototxt             (architecture)
#      synset_words.txt            (1000 class labels)
# =============================================================
#
#  Implement every function below.
#  Complete utils.py first — this file builds on it.
#
#  Self-test:  python model.py
#  Reference:  https://docs.opencv.org/4.x/d6/d0f/group__dnn.html
# =============================================================

import cv2
import numpy as np


def load_labels(filepath: str) -> list:
    """
    Read ImageNet class names from synset_words.txt.

    Each line has the format:
        n01440764 tench, Tinca tinca

    Strip the leading WordNet ID (the first word) and keep only
    the human-readable label. Also strip any surrounding whitespace.

    Args:
        filepath: Path to synset_words.txt.

    Returns:
        List of 1000 strings, e.g. ["tench, Tinca tinca", "goldfish", ...]
    """
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            # Split the line into words and take everything after the first word
            label = ' '.join(line.strip().split()[1:])
            labels.append(label)
    return labels
    raise NotImplementedError


def load_model(prototxt_path: str, caffemodel_path: str):
    """
    Load the GoogLeNet Caffe model into OpenCV's DNN module.

    Two files are required:
        .prototxt    — network architecture definition
        .caffemodel  — trained weights

    Args:
        prototxt_path:   Path to deploy.prototxt
        caffemodel_path: Path to bvlc_googlenet.caffemodel

    Returns:
        A cv2.dnn_Net object.
    """
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    return net
    raise NotImplementedError


def prepare_blob(roi: np.ndarray) -> np.ndarray:
    """
    Convert a color image crop into the blob format GoogLeNet expects.

    GoogLeNet was trained on ImageNet with the following settings:
        - Spatial size:    224 × 224
        - Channel order:   BGR  (no swap needed — OpenCV is already BGR)
        - Mean subtraction: (104, 117, 123) per channel
        - Scale factor:    1.0  (no pixel rescaling)

    Args:
        roi: BGR image crop of any size.

    Returns:
        ndarray of shape (1, 3, 224, 224), dtype float32.
    """
    blob = cv2.dnn.blobFromImage(roi, scalefactor=1.0, size=(224, 224), mean=(104, 117, 123), swapRB=False, crop=False)
    return blob
    raise NotImplementedError


def run_inference(net, blob: np.ndarray) -> np.ndarray:
    """
    Feed a blob into the network and return the raw prediction scores.

    Args:
        net:  Loaded cv2.dnn_Net.
        blob: Prepared blob of shape (1, 3, 224, 224).

    Returns:
        ndarray of shape (1, 1000).
    """
    net.setInput(blob)
    predictions = net.forward()
    return predictions
    raise NotImplementedError


def get_top_prediction(predictions: np.ndarray, labels: list) -> tuple:
    """
    Return the single highest-confidence prediction.

    Args:
        predictions: ndarray of shape (1, 1000).
        labels:      List of 1000 class name strings.

    Returns:
        (label: str, confidence: float)
    """
    # Get the index of the highest confidence score
    top_index = np.argmax(predictions)
    top_label = labels[top_index]
    top_confidence = float(predictions[0, top_index])
    return top_label, top_confidence
    raise NotImplementedError


def get_top_k_predictions(predictions: np.ndarray,
                           labels: list,
                           k: int = 3) -> list:
    """
    Return the top-k predictions sorted by confidence, highest first.

    Args:
        predictions: ndarray of shape (1, 1000).
        labels:      List of 1000 class name strings.
        k:           Number of results to return.

    Returns:
        List of k tuples: [(label: str, confidence: float), ...]
        sorted descending by confidence.
    """
    # Get the indices of the top-k confidence scores
    top_k_indices = np.argsort(predictions[0, :])[::-1][:k]
    # Create a list of (label, confidence) tuples
    top_k_predictions = [(labels[i], float(predictions[0, i])) for i in top_k_indices]
    return top_k_predictions


def draw_prediction(img: np.ndarray,
                    box: tuple,
                    label: str,
                    confidence: float,
                    color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Draw a labeled bounding box onto the image.

    Draw a rectangle around the subject and print
    "label  XX.X%" as text just above the top-left corner.

    Args:
        img:        BGR image to draw on (modified in place).
        box:        (x, y, w, h) in pixels.
        label:      Predicted class name.
        confidence: Float in [0, 1].
        color:      BGR color tuple for box and text. Default green.

    Returns:
        The annotated image.
    """
    x, y, w, h = box
    # Draw the rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    # Prepare the label text
    text = f"{label}  {confidence * 100:.1f}%"
    # Choose a font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    # Get the size of the text
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Draw a filled rectangle behind the text for better visibility
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y), color, cv2.FILLED)
    # Put the text on top of the filled rectangle
    cv2.putText(img, text, (x, y - baseline), font, font_scale, (0, 0, 0), thickness)
    return img
    raise NotImplementedError


# =============================================================
#  SELF-TEST — python model.py
#  All checks must pass before you run main.py.
# =============================================================
if __name__ == "__main__":
    import sys

    print("Testing model.py ...\n")

    try:
        labels = load_labels("synset_words.txt")
        assert len(labels) == 1000, f"Expected 1000, got {len(labels)}"
        assert isinstance(labels[0], str)
        # The WordNet ID should have been stripped
        assert not labels[0].startswith("n0"), \
            f"WordNet ID not stripped — got '{labels[0]}'"
        print(f"  [✓] load_labels           → {len(labels)} labels, first: '{labels[0]}'")
    except Exception as e:
        print(f"  [✗] load_labels           → {e}"); sys.exit(1)

    try:
        net = load_model("deploy.prototxt", "bvlc_googlenet.caffemodel")
        print(f"  [✓] load_model            → {type(net).__name__}")
    except Exception as e:
        print(f"  [✗] load_model            → {e}"); sys.exit(1)

    dummy = np.zeros((300, 300, 3), dtype=np.uint8)
    try:
        blob = prepare_blob(dummy)
        assert blob.shape == (1, 3, 224, 224), f"Wrong shape: {blob.shape}"
        print(f"  [✓] prepare_blob          → shape {blob.shape}")
    except Exception as e:
        print(f"  [✗] prepare_blob          → {e}"); sys.exit(1)

    try:
        preds = run_inference(net, blob)
        assert preds.shape == (1, 1000), f"Expected (1,1000), got {preds.shape}"
        print(f"  [✓] run_inference         → shape {preds.shape}")
    except Exception as e:
        print(f"  [✗] run_inference         → {e}"); sys.exit(1)

    try:
        label, conf = get_top_prediction(preds, labels)
        assert isinstance(label, str)
        assert isinstance(conf, float)
        print(f"  [✓] get_top_prediction    → '{label}'  {conf*100:.1f}%")
    except Exception as e:
        print(f"  [✗] get_top_prediction    → {e}"); sys.exit(1)

    try:
        top3 = get_top_k_predictions(preds, labels, k=3)
        assert len(top3) == 3
        assert top3[0][1] >= top3[1][1] >= top3[2][1], "Must be sorted descending"
        print(f"  [✓] get_top_k_predictions → {[(l[:20], f'{c*100:.1f}%') for l,c in top3]}")
    except Exception as e:
        print(f"  [✗] get_top_k_predictions → {e}"); sys.exit(1)

    try:
        test_img = np.zeros((400, 400, 3), dtype=np.uint8)
        result = draw_prediction(test_img, (50, 50, 200, 200), "test", 0.87)
        assert result is not None and result.shape == test_img.shape
        print(f"  [✓] draw_prediction       → image shape {result.shape}")
    except Exception as e:
        print(f"  [✗] draw_prediction       → {e}"); sys.exit(1)

    print("\n✓ All model.py tests passed — now run: python main.py\n")