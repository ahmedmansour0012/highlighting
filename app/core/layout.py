import torch
import matplotlib.pyplot as plt
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from doclayout_yolo.utils.ops import xyxy2xywh, xywh2xyxy, clip_boxes

import logging, os
logger = logging.getLogger(__name__)


def get_yolo_model_path():
    """
    Returns the path to the YOLO model, downloading it if necessary.
    
    Uses local path if exists, otherwise downloads from Hugging Face Hub.
    """
    model_path = "models/models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"

    logger.debug(f"Checking for existing model at: {model_path}")

    if not os.path.exists(model_path):
        logger.info("Model (doclayout_yolo) not found locally. Downloading from Hugging Face Hub...")
        try:
            model_path = hf_hub_download(
                repo_id="opendatalab/PDF-Extract-Kit-1.0",
                filename="models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt",
                local_dir="./models"
            )
            logger.info(f"Model downloaded and saved at: {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model from Hugging Face Hub: {e}", exc_info=True)
            raise

    else:
        logger.debug(f"Using existing model at: {model_path}")

    return model_path

def layout_detect(images):
    filepath = get_yolo_model_path()
    model = YOLOv10(filepath)
    det_res = model.predict(
      images,   # Image to predict
      imgsz=1024,        # Prediction image size
      conf=0.1,          # Confidence threshold
      device="cpu" ,   # Device to use (e.g., 'cuda:0' or 'cpu')
      )
    return det_res


def get_sorted_detections(det_result):
    """
    Retrieves and sorts detection results by squared distance from the top-left corner.
    """
    detections = det_result.summary(normalize=False)
    if not detections:
        print("No detections found")
        return []

    detections.sort(key=lambda d: (d["box"]["x1"]**2 + d["box"]["y1"]**2))
    return detections




def crop_region(img_array, box, gain=1.01, pad=2):
    """
    Crops a region from a numpy image array given a bounding box.
    Allows optional expansion via gain/pad and conversion to square.

    Args:
        img_array (np.ndarray): The input image as a NumPy array (H, W, C).
        box (dict or list or torch.Tensor): Bounding box in [x1, y1, x2, y2] format.
        gain (float): Multiplier to expand the box size.
        pad (int): Additional pixels to add to width/height.
    Returns:
        np.ndarray: The cropped image.
    """
    # Convert box to tensor if needed
    if not isinstance(box, torch.Tensor):
        if isinstance(box, dict):
            x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
        else:
            x1, y1, x2, y2 = map(int, box)
        box = torch.tensor([[x1, y1, (x2), y2]])

    # Convert xyxy -> xywh
    b = xyxy2xywh(box.view(-1, 4))

    # Apply gain and padding
    b[:, 2:] = b[:, 2:] * gain + pad

    # Back to xyxy
    xyxy = xywh2xyxy(b).long()

    # Clip coordinates to image bounds
    xyxy = clip_boxes(xyxy, img_array.shape)

    # Crop image
    crop = img_array[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2])]

    # Add black line at the top
    crop[0, :] = 0

    # crop.save(f"header_section.png")
    return crop
