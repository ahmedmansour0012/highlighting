# Table Processing Pipeline

from PIL import Image
from numpy import asarray
from surya.table_rec import TableRecPredictor

def get_table_predictions(table_image):
  table_rec_predictor = TableRecPredictor()
  table_predictions = table_rec_predictor([table_image])
  return table_predictions


def col_crops(image, coL_list, by_col, padding=1, gain=1.02):
    """
    Crops the image using bounding boxes adjusted by padding and gain.
    """
    cropped = []
    coords = []
    upscale = 2
    if by_col:
        for idx, (bbox, col_id) in enumerate(coL_list):
            x_min, y_min, x_max, y_max = map(int, bbox)

            # Step 1: Apply padding
            x_min_pad = x_min - padding
            y_min_pad = y_min - padding
            x_max_pad = x_max + padding
            y_max_pad = y_max + padding

            # Step 2: Apply gain scaling
            width = x_max_pad - x_min_pad
            height = y_max_pad - y_min_pad

            center_x = (x_min_pad + x_max_pad) / 2
            center_y = (y_min_pad + y_max_pad) / 2

            new_width = width * gain
            new_height = height * gain

            # Step 3: Compute adjusted coordinates
            x_min_adj = int(round(center_x - new_width / 2))
            x_max_adj = int(round(center_x + new_width / 2))
            y_min_adj = int(round(center_y - new_height / 2))
            y_max_adj = int(round(center_y + new_height / 2))

            # Step 4: Crop the image using adjusted coordinates
            im1 = image.crop((x_min_adj, y_min_adj, x_max_adj, y_max_adj))
            im = im1.resize((im1.width * upscale, im1.height * upscale), Image.LANCZOS)

            coords.append((x_min_adj*upscale, y_min_adj*upscale, x_max_adj*upscale, y_max_adj*upscale))

            cropped.append(asarray(im))
        return cropped ,coords
    else:
        for col in coL_list:
            x_min, y_min, x_max, y_max = map(int, col.bbox)

            # Step 1: Apply padding
            x_min_pad = x_min - padding
            y_min_pad = y_min - padding
            x_max_pad = x_max + padding
            y_max_pad = y_max + padding

            # Step 2: Apply gain scaling
            width = x_max_pad - x_min_pad
            height = y_max_pad - y_min_pad

            center_x = (x_min_pad + x_max_pad) / 2
            center_y = (y_min_pad + y_max_pad) / 2

            new_width = width * gain
            new_height = height * gain

            # Step 3: Compute adjusted coordinates
            x_min_adj = int(round(center_x - new_width / 2))
            x_max_adj = int(round(center_x + new_width / 2))
            y_min_adj = int(round(center_y - new_height / 2))
            y_max_adj = int(round(center_y + new_height / 2))

            # Step 4: Crop the image using adjusted coordinates
            im1 = image.crop((x_min_adj, y_min_adj, x_max_adj, y_max_adj))
            im = im1.resize((im1.width * 2, im1.height * 2), Image.LANCZOS)
            coords.append((x_min_adj*upscale, y_min_adj*upscale, x_max_adj*upscale, y_max_adj*upscale))

            cropped.append(asarray(im))
        return cropped ,coords






def process_table_image(image,ocr):
    """
    Main function to process a table image, extract headers, and save them.
    """
    img = Image.fromarray(image)
    # im = upscale_and_center_image(img)
    im = img.resize((img.width *2 , img.height * 2), Image.LANCZOS) # LANCZOS for high-quality downsampling/upsampling
    # im = img
    table_predictions = get_table_predictions(im)
    columns,by_col = get_col_region(table_predictions, ocr, im)

    col_crop,coords = col_crops(im, columns, by_col)

    return col_crop,coords

def get_header_row_id(cells,ocr,image):
  val = -1
  # print(cells)
  for idx, cell in enumerate(cells):
    if val == cell.row_id:
      continue
    else:
      # print(idx)
      x_min, y_min, x_max, y_max = map(int, cell.bbox)
      im1 = image.crop((x_min, y_min, x_max, y_max))
      try:
        x_min, y_min, x_max, y_max = map(int, cells[idx+1].bbox)
        im2 = image.crop((x_min, y_min, x_max, y_max))
        text2 = ocr.predict(asarray(im2))[0]['rec_texts']
      except:
          text2 =[None]

      res = ocr.predict(asarray(im1))
      print(res)
      text = res[0]['rec_texts']
      if text is not None:
        # print(cell.row_id)
        return cell.row_id
      if text2[0] is not None:
        # print(cell.row_id)
        return cell.row_id
      else:
        val = cell.row_id

def extend_cell_to_image_end(cell, image_height):
    # Update polygon
    cell.polygon[2][1] = image_height  # top-right y-coordinate
    cell.polygon[3][1] = image_height  # bottom-right yffffffffffffffffff-coordinate

    # Update bounding box
    cell.bbox[3] = image_height  # max y-coordinate

    return cell

def get_col_region(table_predictions,ocr,image):
  image_height = image.height
  cells = table_predictions[0].cells
  col_list = []
  # row_one = []
  header_row_id = get_header_row_id(cells,ocr,image)
  # Collect header cells
  header_cells = [cell for cell in cells if cell.row_id == header_row_id]
  # Check if any cell has colspan > 3
  has_large_colspan = any(cell.colspan > 3 for cell in header_cells)
  if has_large_colspan:
    columns = table_predictions[0].cols
    col_list = extract_header_regions(columns)
    by_col = True
  else:
    by_col = False
    for cell in cells:
      if cell.row_id == header_row_id:
        col = extend_cell_to_image_end(cell, image_height)
        col_list.append(col)
  return col_list, by_col

def extract_header_regions(columns):
    final_crops = []
    i = 0
    # columns = table_predictions[0].cols
    while i < len(columns):
      if not (i+1)>=len(columns):
        if columns[i].is_header == True and columns[i + 1].is_header ==False:
              # Merge current and next column
              col1 = columns[i]
              col2 = columns[i + 1]

              merged_bbox = [
                  col1.bbox[0],  # x1 of first column
                  min(col1.bbox[1], col2.bbox[1]),  # min y1
                  col2.bbox[2],  # x2 of second column
                  max(col1.bbox[3], col2.bbox[3])   # max y2
              ]

              # Use the first column's col_id for naming
              final_crops.append((merged_bbox, col1.col_id))
              i += 2  # Skip both columns
        elif columns[i].is_header == False and columns[i + 1].is_header ==False:
              col1 = columns[i]
              col2 = columns[i + 1]
              final_crops.append((col1.bbox, col1.col_id))
              final_crops.append((col2.bbox, col2.col_id))
              i += 2  # Skip both columns
        else:
              final_crops.append((columns[i].bbox, columns[i].col_id))
              i += 1
      else:
          col1 = columns[i]
          final_crops.append((col1.bbox, col1.col_id))
          i+=1
    return final_crops
