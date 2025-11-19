from PIL import Image
from surya.layout import LayoutPredictor

# Initialize once (can be reused across calls)
layout_predictor = LayoutPredictor()

def istable(image_arr) -> bool:
    """
    Detect whether the given image contains a table
    using Surya's LayoutPredictor.

    Args:
        image_arr: A NumPy array or PIL.Image representing the image.

    Returns:
        bool: True if a table-like region is detected, False otherwise.
    """
    # Ensure PIL Image
    if not isinstance(image_arr, Image.Image):
        image = Image.fromarray(image_arr)
    else:
        image = image_arr

    # Run prediction
    layout_predictions = layout_predictor([image])

    # Each prediction has .bboxes with .label
    prediction = layout_predictions[0]

    # Check if any detected object is labeled "table"
    labels = [bbox.label.lower() for bbox in prediction.bboxes]
    return any("table" in lbl for lbl in labels)
