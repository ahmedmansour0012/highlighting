import numpy as np
from collections import defaultdict

# ---------- PARAMETERS ----------
V_OVERLAP_THRESH = 0.5
H_GAP_MULT = 1.5
PARA_GAP_MULT = 1.5
INDENT_THRESH = 30
LINE_INDENT_THRESH = 20
X_GAP_MULT = 0.8  # for column clustering
CELL_GAP_MULT = 0.5  # for splitting wide lines into cells
# --------------------------------

def boxes_from_rec(rec_boxes):
    boxes = []
    for b in rec_boxes:
        b = np.asarray(b).astype(float)
        if b.size == 4:
            xmin, ymin, xmax, ymax = b
        elif b.size == 8:
            xs = b[0::2]; ys = b[1::2]
            xmin, ymin, xmax, ymax = xs.min(), ys.min(), xs.max(), ys.max()
        else:
            raise ValueError(f"rec_boxes element has unexpected length: {b.size}")
        boxes.append([xmin, ymin, xmax, ymax])
    return np.array(boxes)

# # ---------- Column clustering ----------
# def cluster_tokens_by_x(boxes, tokens, x_gap_mult=X_GAP_MULT):
#     widths = boxes[:,2] - boxes[:,0]
#     median_width = np.median(widths) if len(widths) > 0 else 1.0
#     gap_thresh = median_width * x_gap_mult

#     sorted_idx = np.argsort(boxes[:,0])
#     columns = []
#     cur_col = [sorted_idx[0]]

#     for idx in sorted_idx[1:]:
#         prev_idx = cur_col[-1]
#         gap = boxes[idx,0] - boxes[prev_idx,2]
#         if gap > gap_thresh:
#             columns.append(cur_col)
#             cur_col = [idx]
#         else:
#             cur_col.append(idx)
#     columns.append(cur_col)

#     token_columns = []
#     for col in columns:
#         token_columns.append([tokens[i] for i in col])
#     return token_columns, columns

import numpy as np

# ---------- Column clustering ----------
def cluster_tokens_by_x(boxes, tokens, x_gap_mult=X_GAP_MULT):
    # Convert to numpy array
    boxes = np.array(boxes, dtype=float)

    # Handle empty boxes
    if boxes.size == 0:
        return [], []

    # Ensure 2D shape (n,4)
    if boxes.ndim == 1:
        if boxes.size % 4 != 0:
            raise ValueError(f"Unexpected boxes shape {boxes.shape}, expected multiple of 4 values")
        boxes = boxes.reshape(-1, 4)

    widths = boxes[:, 2] - boxes[:, 0]
    median_width = np.median(widths) if len(widths) > 0 else 1.0
    gap_thresh = median_width * x_gap_mult

    # Sort left-to-right
    sorted_idx = np.argsort(boxes[:, 0])
    columns = []
    cur_col = [sorted_idx[0]]

    for idx in sorted_idx[1:]:
        prev_idx = cur_col[-1]
        gap = boxes[idx, 0] - boxes[prev_idx, 2]
        if gap > gap_thresh:
            columns.append(cur_col)
            cur_col = [idx]
        else:
            cur_col.append(idx)
    columns.append(cur_col)

    # Assign tokens to columns
    token_columns = [[tokens[i] for i in col] for col in columns]

    return token_columns, columns


# ---------- Vertical grouping ----------
def union_find_groups(boxes, v_overlap_thresh=V_OVERLAP_THRESH):
    n = len(boxes)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    ymin = boxes[:,1]; ymax = boxes[:,3]; heights = ymax - ymin
    for i in range(n):
        for j in range(i+1, n):
            overlap = min(ymax[i], ymax[j]) - max(ymin[i], ymin[j])
            if overlap <= 0: continue
            min_h = min(heights[i], heights[j])
            if (overlap / (min_h + 1e-9)) >= v_overlap_thresh:
                union(i,j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())

def split_group_into_lines(group_indices, boxes, median_width):
    idx_sorted = sorted(group_indices, key=lambda i: boxes[i,0])
    gap_thresh = max(median_width * H_GAP_MULT, 10)
    lines, cur = [], [idx_sorted[0]]
    for idx in idx_sorted[1:]:
        prev = cur[-1]
        gap = boxes[idx,0] - boxes[prev,2]
        if gap > gap_thresh:
            lines.append(cur); cur = [idx]
        else:
            cur.append(idx)
    lines.append(cur)
    return lines

def compute_line_bbox(indices, boxes):
    xs_min = boxes[indices,0].min()
    ys_min = boxes[indices,1].min()
    xs_max = boxes[indices,2].max()
    ys_max = boxes[indices,3].max()
    return [xs_min, ys_min, xs_max, ys_max]

# ---------- Lines per column ----------
def lines_from_boxes_by_column(boxes, col_indices):
    median_width = np.median((boxes[:,2]-boxes[:,0]) + 1e-9)
    col_boxes = boxes[col_indices]
    v_groups = union_find_groups(col_boxes, v_overlap_thresh=V_OVERLAP_THRESH)
    all_lines = []

    for g in v_groups:
        g_orig = [col_indices[i] for i in g]
        sublines = split_group_into_lines(g_orig, boxes, median_width)
        for sub in sublines:
            bbox = compute_line_bbox(sub, boxes)
            all_lines.append({"indices": sub, "bbox": bbox})
    all_lines = sorted(all_lines, key=lambda L: L["bbox"][1])
    return all_lines

# ---------- Paragraph & wrapped line ----------
def paragraphs_from_lines(lines):
    heights = np.array([L["bbox"][3] - L["bbox"][1] for L in lines])
    median_line_h = np.median(heights) if len(heights)>0 else 1.0
    para_gap_thresh = PARA_GAP_MULT * median_line_h
    paragraphs, cur_para = [], [lines[0]] if lines else []
    for prev, curr in zip(lines, lines[1:]):
        prev_bbox, curr_bbox = prev["bbox"], curr["bbox"]
        vertical_gap = curr_bbox[1] - prev_bbox[3]
        left_diff = abs(curr_bbox[0] - prev_bbox[0])
        if vertical_gap <= para_gap_thresh and left_diff <= max(INDENT_THRESH, 0.2 * prev_bbox[2]):
            cur_para.append(curr)
        else:
            paragraphs.append(cur_para); cur_para = [curr]
    if cur_para: paragraphs.append(cur_para)
    return paragraphs

def merge_wrapped_lines(para, indent_thresh=LINE_INDENT_THRESH):
    merged = []
    i = 0
    while i < len(para):
        line = para[i]
        text = line["text_joined"]
        bbox = line["bbox"]

        j = i + 1
        while j < len(para):
            next_line = para[j]
            next_text = next_line["text_joined"].strip()
            left_diff = abs(next_line["bbox"][0] - bbox[0]) < indent_thresh
            starts_lower = next_text and next_text[0].islower()
            starts_connector = any(next_text.lower().startswith(w) for w in ["and","or","with","of","to","for"])
            words = next_text.split()
            looks_like_item = (
                bool(words) and (
                    words[0].isupper() or
                    words[0][0].isdigit() or
                    words[0].endswith("=")
                )
            )
            if (left_diff and (starts_lower or starts_connector)) and not looks_like_item:
                text += " " + next_text
                bbox = [
                    min(bbox[0], next_line["bbox"][0]),
                    min(bbox[1], next_line["bbox"][1]),
                    max(bbox[2], next_line["bbox"][2]),
                    max(bbox[3], next_line["bbox"][3])
                ]
                j += 1
            else:
                break

        merged.append({"text": text, "bbox": bbox, "indices": line["indices"]})
        i = j
    return merged

# ---------- Split wide lines into cells ----------
def split_line_into_cells(tokens, token_boxes, line_indices, line_bbox, gap_mult=CELL_GAP_MULT):
    if len(line_indices) <= 1:
        return [{"text": tokens[line_indices[0]], "bbox": line_bbox}]

    # Sort tokens left-to-right
    idx_sorted = sorted(line_indices, key=lambda i: token_boxes[i][0])
    widths = [token_boxes[i][2]-token_boxes[i][0] for i in idx_sorted]
    median_width = np.median(widths)
    gap_thresh = median_width * gap_mult

    cells = []
    cur_tokens = [tokens[idx_sorted[0]]]
    cur_bbox = token_boxes[idx_sorted[0]].copy()

    for i in idx_sorted[1:]:
        bbox = token_boxes[i]
        gap = bbox[0] - cur_bbox[2]
        if gap > gap_thresh:
            cells.append({"text": " ".join(cur_tokens), "bbox": cur_bbox})
            cur_tokens = [tokens[i]]
            cur_bbox = bbox.copy()
        else:
            cur_tokens.append(tokens[i])
            cur_bbox = [
                min(cur_bbox[0], bbox[0]),
                min(cur_bbox[1], bbox[1]),
                max(cur_bbox[2], bbox[2]),
                max(cur_bbox[3], bbox[3])
            ]
    cells.append({"text": " ".join(cur_tokens), "bbox": cur_bbox})
    return cells

# ---------- Full column-aware pipeline ----------
def cluster_paragraphs_from_rec_columns(rec_boxes, rec_texts):
    boxes = boxes_from_rec(rec_boxes)
    token_columns, col_indices_list = cluster_tokens_by_x(boxes, rec_texts)

    # Sort columns left-to-right
    columns_sorted = sorted(zip(token_columns, col_indices_list),
                            key=lambda x: min(boxes[i,0] for i in x[1]))

    all_lines = []
    for col_tokens, col_indices in columns_sorted:
        col_lines = lines_from_boxes_by_column(boxes, col_indices)
        for L in col_lines:
            indices = sorted(L["indices"], key=lambda i: boxes[i,0])
            L["texts"] = [rec_texts[i] for i in indices]
            L["text_joined"] = " ".join(L["texts"])
            L["bbox"] = compute_line_bbox(indices, boxes)
        all_lines.extend(col_lines)

    # sort all lines top-to-bottom
    all_lines = sorted(all_lines, key=lambda L: L["bbox"][1])
    paragraphs = paragraphs_from_lines(all_lines)

    para_outputs = []
    new_rec_boxes = []
    new_rec_texts = []

    for p_idx, para in enumerate(paragraphs):
        merged_lines = merge_wrapped_lines(para)
        # Split wide lines into cells
        for L in merged_lines:
            cells = split_line_into_cells(rec_texts, boxes, L["indices"], L["bbox"])
            for cell in cells:
                new_rec_texts.append(cell["text"])
                new_rec_boxes.append(cell["bbox"])
        para_outputs.append({"id": p_idx, "texts": [t["text"] for t in merged_lines],
                             "bboxes": [L["bbox"] for L in merged_lines]})

    new_rec_boxes = np.array(new_rec_boxes, dtype=np.int32)
    return para_outputs, new_rec_boxes, new_rec_texts
