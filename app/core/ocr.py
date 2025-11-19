import cv2
import logging
from paddleocr import PaddleOCR
from .layout import get_sorted_detections, crop_region  

logger = logging.getLogger(__name__)


_ocr_instance = None

def get_ocr_instance():
    """
    Returns the global OCR instance, initializing it if necessary.
    """
    global _ocr_instance
    if _ocr_instance is None:
        logger.info("Initializing PaddleOCR model...")
        _ocr_instance = PaddleOCR(
            lang="en",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=640,
            text_recognition_batch_size=16,
        )
    return _ocr_instance

import numpy as np

def get_ocr_object_per_page(images,ocr,rec_res=None):
  if rec_res:  
    ocr_results =[]
    ocr_results_per_page =[]

    for i,image in enumerate(images):
        detections = get_sorted_detections(rec_res[i])
        if not detections:
            return
        for idx, det in enumerate(detections):
            if det['name']=='table' or det['name']=='figure':
                continue
            else:
                img_array = np.array(image)
                cropped = crop_region(img_array, det["box"])
                h, w = cropped.shape[:2]
                cropped = cv2.resize(cropped, (int(w * 2.0), int(h * 2.0)), interpolation=cv2.INTER_CUBIC)

                resultf = ocr.predict(cropped)
                ocr_results_per_page.append(resultf)
        ocr_results.append(ocr_results_per_page)
        ocr_results_per_page=[]
  else:  
    ocr_results =[]
    for image in images:
        np_img = np.asarray(image)
        res = ocr.predict(np_img)
        if res:
            ocr_results.append(res)

  return ocr_results

def join_ocr_texts(rec_texts, rec_polys, y_tolerance=12, x_tolerance=20, paragraph_gap=40):
    """
    Join OCR texts into structured text using both X and Y positions.

    Args:
        rec_texts (list[str]): recognized words.
        rec_polys (list[list]): bounding boxes, each is list of 4 (x,y).
        y_tolerance (int): max vertical difference for same line.
        x_tolerance (int): min gap between words before inserting extra space.
        paragraph_gap (int): vertical gap that creates blank line between paragraphs.

    Returns:
        str: reconstructed text with structure.
    """
    if not rec_texts or not rec_polys:
        return ""

    # collect word centers + bbox
    words = []
    for text, poly in zip(rec_texts, rec_polys):
        if not text.strip():
            continue
        poly = np.array(poly)
        x_min, y_min = poly[:,0].min(), poly[:,1].min()
        x_max, y_max = poly[:,0].max(), poly[:,1].max()
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        words.append((text, cx, cy, x_min, x_max, y_min, y_max))

    # sort top to bottom, then left to right
    words.sort(key=lambda x: (x[2], x[1]))

    lines = []
    current_line = []
    last_y = None

    for word in words:
        text, cx, cy, x_min, x_max, y_min, y_max = word

        if last_y is None or abs(cy - last_y) <= y_tolerance:
            current_line.append(word)
            last_y = cy if last_y is None else (last_y + cy) / 2
        else:
            # flush current line
            current_line.sort(key=lambda w: w[3])  # sort by left edge
            line_str = current_line[0][0]
            for i in range(1, len(current_line)):
                prev = current_line[i-1]
                cur = current_line[i]
                # insert extra space if big horizontal gap
                if cur[3] - prev[4] > x_tolerance:
                    line_str += "   " + cur[0]
                else:
                    line_str += " " + cur[0]
            lines.append((last_y, line_str))

            # check for paragraph break
            if cy - last_y > paragraph_gap:
                lines.append((cy - 0.1, ""))  # blank line marker

            current_line = [word]
            last_y = cy

    # flush last line
    if current_line:
        current_line.sort(key=lambda w: w[3])
        line_str = current_line[0][0]
        for i in range(1, len(current_line)):
            prev = current_line[i-1]
            cur = current_line[i]
            if cur[3] - prev[4] > x_tolerance:
                line_str += "   " + cur[0]
            else:
                line_str += " " + cur[0]
        lines.append((last_y, line_str))

    # return sorted by Y
    return "\n".join(l for _, l in sorted(lines, key=lambda x: x[0]))
