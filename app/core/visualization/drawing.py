from PIL import ImageDraw

## Result Mapping and Visualization

import numpy as np

def map_matches_to_original_image(all_hits: list):
    """
    Map matched hits back to original image coordinates.
    Works on page → tables → hits structure.
    Adds padding (2% top, 2% right, 3% left),
    clipped to column boundaries.
    """
    mapped_boxes = []

    for page_result in all_hits:
        page_idx = page_result["page_index"]

        for table_result in page_result["tables"]:
            table = table_result["table"]
            hits = table_result["hits"]

            for match in hits:
                tb_x1, tb_y1 = float(match['table_bbox']['x1']), float(match['table_bbox']['y1'])
                col_idx = match['column_index']
                col_bbox_list = match['column_bbox']

                if not col_bbox_list or len(col_bbox_list) <= col_idx:
                    continue

                # column bounding box (scaled)
                col_bbox = col_bbox_list[col_idx]
                cb_x1, cb_y1, cb_x2, cb_y2 = [v / 4 for v in col_bbox]

                word_box_local = np.array(match['word_bbox'])
                word_box_scaled = (word_box_local / 4).astype(int)

                abs_x1 = word_box_scaled[0] + cb_x1 + tb_x1
                abs_y1 = word_box_scaled[1] + cb_y1 + tb_y1
                abs_x2 = word_box_scaled[2] + cb_x1 + tb_x1
                abs_y2 = word_box_scaled[3] + cb_y1 + tb_y1

                # ---- Add padding ----
                width = abs_x2 - abs_x1
                height = abs_y2 - abs_y1

                abs_y1 -= 0.02 * width     # top padding (2% of width)
                abs_x2 += 0.02 * height    # right padding (2% of height)
                abs_x1 -= 0.60 * height    # left padding (3% of height)

                # ---- Clip to column bounds ----
                abs_x1 = max(abs_x1, cb_x1 + tb_x1)  # left bound
                abs_y1 = max(abs_y1, cb_y1 + tb_y1)  # top bound
                abs_x2 = min(abs_x2, cb_x2 + tb_x1)  # right bound
                abs_y2 = min(abs_y2, cb_y2 + tb_y1)  # bottom bound (safety)

                mapped_boxes.append({
                    "page_index": page_idx,
                    "table": table,
                    "term": match['term'],
                    "orig_box": [abs_x1, abs_y1, abs_x2, abs_y2],
                    "local_box": word_box_scaled.tolist(),
                    "column_index": col_idx
                })

    return mapped_boxes

def draw_mapped_boxes(images, mapped_boxes, out_prefix="page"):
    """
    Draw mapped boxes onto their corresponding page images.

    Args:
        images: list of PIL.Image objects (indexed by page)
        mapped_boxes: list of dicts with 'page_index' and 'orig_box'
        out_prefix: prefix for saving output images

    Returns:
        tuple: (
            list_of_page_indices: sorted list of page indices that had boxes drawn,
            list_of_annotated_images: list of PIL.Image objects (clean, fully loaded),
            list_of_saved_image_paths: full file paths of saved PNG images (optional)
        )
    """
    # Group boxes by page
    page_to_boxes = {}
    for box_info in mapped_boxes:
        page_idx = box_info["page_index"]
        if page_idx < 0 or page_idx >= len(images):
            print(f"Warning: page_index {page_idx} is out of bounds (total pages: {len(images)}), skipping.")
            continue
        page_to_boxes.setdefault(page_idx, []).append(box_info)

    annotated_images = []   # List of PIL Image objects (in-page-order)
    saved_image_paths = []  # Optional: paths to saved files
    processed_page_indices = []

    # Process each page that has boxes, in sorted order
    for page_idx in sorted(page_to_boxes.keys()):
        # Work on a copy of the specific image to avoid modifying original
        img = images[page_idx].copy()
        draw = ImageDraw.Draw(img)

        for box_info in page_to_boxes[page_idx]:
            x1, y1, x2, y2 = map(int, box_info["orig_box"])
            draw.rectangle([x1, y1, x2, y2], outline="Red", width=3)

        # Save to disk (optional, for debugging or external use)
        output_path = f"{out_prefix}_{page_idx}.png"
        img.save(output_path)
        saved_image_paths.append(output_path)

        # Store clean, fully-loaded image object (no file handle dependency)
        # Force decode and ensure RGB mode for safe PDF export later
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.load()  # Ensure pixel data is loaded into memory
        annotated_images.append(img)
        processed_page_indices.append(page_idx)

    print(f"Saved {len(saved_image_paths)} annotated pages: {saved_image_paths}")

    return processed_page_indices, annotated_images, saved_image_paths
# mapped_boxes = map_matches_to_original_image(all_hits)

# draw_mapped_boxes(images, mapped_boxes, out_prefix="page")




