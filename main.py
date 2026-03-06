# =============================================================
#  main.py  —  CV Workshop: Full Pipeline Orchestrator
#  Part 3 of 3
# =============================================================
#
#  Implement run_pipeline(), run_batch(), and Task 3 below.
#  Import and use functions from utils.py and model.py.
#  Do not add cv2 calls anywhere except the display step.
#
#  Prerequisites:
#      python utils.py   → all ✓ before starting here
#      python model.py   → all ✓ before starting here
#
#  Run:
#      python main.py
#      python main.py --image images/cat.jpg --label cat
#      python main.py --batch images/
# =============================================================

import cv2
import numpy as np
import argparse
import os

from utils import load_image, preprocess, find_subject_contour, crop_roi
from model import (load_labels, load_model, prepare_blob,
                   run_inference, get_top_prediction,
                   get_top_k_predictions, draw_prediction)

# ── Change these to classify something different ───────────
IMAGE_PATH      = "images/dog.jpg"
TARGET_LABEL    = "dog"
MODEL_PROTOTXT  = "deploy.prototxt"
MODEL_WEIGHTS   = "bvlc_googlenet.caffemodel"
LABELS_FILE     = "synset_words.txt"


# =============================================================
#  TASK 1 — implement run_pipeline()
#
#  Wire the complete pipeline using only the imported functions.
#  The steps, in order:
#
#    1. Load the image
#    2. Load the 1000 class labels
#    3. Load the DNN model
#    4. Preprocess the image (grayscale → blur → edges)
#    5. Find the largest subject contour
#       → if none found: print a message and return None
#    6. Crop the ROI from the color image
#    7. Prepare the blob
#    8. Run inference
#    9. Get the top-1 prediction
#   10. If target_label was provided, print whether it matched
#   11. Draw the bounding box and label on the image
#   12. Display the result with cv2.imshow / cv2.waitKey(0)
#   13. Return (label, confidence)
# =============================================================

def run_pipeline(image_path: str, target_label: str = "") -> tuple:
    """
    Run the full detection and classification pipeline on one image.

    Args:
        image_path:   Path to input image.
        target_label: Optional expected label for match reporting.

    Returns:
        (label, confidence) or None if no subject was found.
    """
    # Step 1: Load the image
    img = load_image(image_path)
    # Step 2: Load the 1000 class labels
    labels = load_labels(LABELS_FILE)
    # Step 3: Load the DNN model
    net = load_model(MODEL_PROTOTXT, MODEL_WEIGHTS)
    # Step 4: Preprocess the image (grayscale → blur → edges)
    edges = preprocess(img)
    # Step 5: Find the largest subject contour
    contour = find_subject_contour(edges)
    if contour is None:
        print(f"[!] No subject contour found in '{image_path}'.")
        return None
    # Step 6: Crop the ROI from the color image
    roi, box = crop_roi(img, contour)
    # Step 7: Prepare the blob
    blob = prepare_blob(roi)
    # Step 8: Run inference
    predictions = run_inference(net, blob)
    # Step 9: Get the top-1 prediction
    label, confidence = get_top_prediction(predictions, labels)
    # Step 10: If target_label was provided, print whether it matched
    if target_label:
        match_status = "MATCH" if label.lower() == target_label.lower() else "NO MATCH"
        print(f"Target label: '{target_label}' → {match_status}")
    # Step 11: Draw the bounding box and label on the image
    draw_prediction(img, box, label, confidence)
    # Step 12: Display the result with cv2.imshow / cv2.waitKey(0)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Step 13: Return (label, confidence)
    return label, confidence
    raise NotImplementedError


# =============================================================
#  TASK 2 — implement run_batch()
#
#  Classify every .jpg and .png in a folder.
#
#  Requirements:
#    - Load the labels and model ONCE, outside the loop
#    - Use get_top_k_predictions() to print the top 3 per image
#    - Skip images where find_subject_contour() returns None
#    - Catch FileNotFoundError per image — print it and continue
#    - After processing all images, print a summary:
#        · total images processed (excluding skipped)
#        · how many had top-confidence >= 70%
#        · the single best prediction (label, confidence, filename)
# =============================================================

def run_batch(folder: str) -> None:
    """
    Classify every .jpg / .png in a folder and print a summary.

    Args:
        folder: Path to directory containing images.
    """
    labels = load_labels(LABELS_FILE)
    net = load_model(MODEL_PROTOTXT, MODEL_WEIGHTS)
    total_processed = 0
    high_confidence_count = 0
    best_prediction = ("", 0.0, "")  # (label, confidence, filename)
    for filename in os.listdir(folder):
        if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
            continue
        image_path = os.path.join(folder, filename)
        try:
            img = load_image(image_path)
            edges = preprocess(img)
            contour = find_subject_contour(edges)
            if contour is None:
                print(f"[!] No subject contour found in '{filename}'. Skipping.")
                continue
            roi, box = crop_roi(img, contour)
            blob = prepare_blob(roi)
            predictions = run_inference(net, blob)
            top_k = get_top_k_predictions(predictions, labels, k=3)
            print(f"Top 3 predictions for '{filename}':")
            for label, confidence in top_k:
                print(f"  {label}: {confidence*100:.2f}%")
            total_processed += 1
            if top_k[0][1] >= 0.7:
                high_confidence_count += 1
            if top_k[0][1] > best_prediction[1]:
                best_prediction = (top_k[0][0], top_k[0][1], filename)
        except FileNotFoundError:
            print(f"[!] File not found: '{filename}'. Skipping.")
        except Exception as e:
            print(f"[!] Error processing '{filename}': {e}. Skipping.")
    print("\nBatch Summary:")
    print(f"  Total images processed: {total_processed}")
    print(f"  Images with top-confidence >= 70%: {high_confidence_count}")
    if best_prediction[1] > 0:
        print(f"  Best prediction: '{best_prediction[0]}' with {best_prediction[1]*100:.2f}% in '{best_prediction[2]}'")
    raise NotImplementedError


# =============================================================
#  TASK 3 (STRETCH) — Confidence threshold filter
#
#  Add a min_confidence parameter (float, default 0.5) to
#  run_pipeline(). When the top prediction falls below the
#  threshold:
#    - Draw the label in RED instead of green
#    - Prefix the label text with "Low confidence: "
#    - Return ("uncertain", confidence) instead
#
#  Propagate min_confidence through run_batch() as well.
#
#  You will need to revisit draw_prediction() in model.py
#  to support a custom color argument — it already has one
#  in the signature, make sure your implementation uses it.
# =============================================================


# =============================================================
#  CLI — do not modify
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV Workshop Classifier")
    parser.add_argument("--image", type=str, default=IMAGE_PATH)
    parser.add_argument("--label", type=str, default=TARGET_LABEL)
    parser.add_argument("--batch", type=str, default="")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch)
    else:
        result = run_pipeline(args.image, args.label)
        if result:
            label, conf = result
            print(f"Final: {label}  ({conf*100:.1f}%)")