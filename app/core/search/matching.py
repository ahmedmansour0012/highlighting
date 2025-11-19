import re
from collections import deque
from typing import Dict, List, Tuple, Any
from core.serial_extraction.parsing import _split_serial

def flatten(xss):
    return [x for xs in xss for x in xs]
def correct_ocr_text(text):
    # Only fix likely OCR mistakes: letters → digits
    return text.translate(str.maketrans({
        'o': '0',
        'O': '0',
        # 'l': '1',
        # '1': 'l',
        # 'I': '1',   # if your OCR confuses capital I with 1
        # 'i': '1', # optional, but often overcorrection
    }))


def exact_match_in_tables(tables: List, serial_no: str):
    found = []
    new_splits = []
    total_col_count = 0
    new = []
    # keep all parts in order
    parts = _split_serial(serial_no)
    # use casefold lookup but preserve duplicates/order in deque
    parts_lookup = deque(parts)
    # print(parts_lookup)
    for i, part in enumerate(parts_lookup):
      if any(c in part for c in 'o0O'):  # include uppercase if needed
          corrected_part = correct_ocr_text(part)
          new.append(corrected_part)
    for n in new:
      parts_lookup.append(n)

    pages: Dict[int, Dict[str, Any]] = {}
    big_text = ""  # start empty before loop
    for table in tables:
        col_count = 0
        page_idx = table.page_index
        if page_idx not in pages:
            pages[page_idx] = {"page_index": page_idx, "tables": []}

        table_hits = []
        for col_idx, col in enumerate(table.columns):
            big_text += " " + " ".join(col.texts).lower()
            for word_idx, word in enumerate(col.texts):
                cleaned = re.sub(r'[^a-zA-Z0-9_=/-]', ' ', word)
                tokens = cleaned.split('=')

                word_bbox = col.boxes[word_idx]

                for token in tokens:
                    key = token.strip().casefold()
                    # check if next expected part matches
                    for i, part in enumerate(parts_lookup):
                        # print(part,type(part))
                        # print(key,type(key))
                        if key == part.strip().casefold():
                            # print(key, part)
                            original_part = part
                            found.append(token)
                            # remove this part so it can’t match again
                            parts_lookup.remove(part)
                            if not col_count:
                              col_count = table.num_columns
                              # print(col_count)

                            table_hits.append({
                                'tableclass': table,
                                "page_index": table.page_index,
                                "table_bbox": table.table_bbox,
                                "column_index": col_idx,
                                "column_bbox": getattr(col, "bbox", None),
                                "matched_text": token,
                                "term": original_part,
                                "word_bbox": word_bbox,
                            })
                            break  # stop checking this token once matched

        pages[page_idx]["tables"].append({"table": table, "hits": table_hits})
        total_col_count = total_col_count + col_count


    not_found = list(parts_lookup)
    # print(not_found)
    # Fallback: look for not_found parts as substrings in full cell texts
    if not_found:
        substring_hits, substring_found, not_found = find_substring_matches_in_tables(tables, not_found)
        # Add these hits to your all_hits structure
        # Group by page like you do elsewhere
        for hit in substring_hits:
            page_idx = hit["page_index"]
            if page_idx not in pages:
                pages[page_idx] = {"page_index": page_idx, "tables": []}

            # Find or create table entry in pages[page_idx]["tables"]
            table_found = False
            for tbl_entry in pages[page_idx]["tables"]:
                if tbl_entry["table"] is hit["tableclass"]:
                    tbl_entry["hits"].append(hit)
                    table_found = True
                    break
            if not table_found:
                pages[page_idx]["tables"].append({"table": hit["tableclass"], "hits": [hit]})

        found.extend(substring_found)
    all_hits = list(pages.values())

    # print(big_text)


    all_hits, found, not_found = find_combined_parts_in_tables(all_hits,found, not_found)

    for serial in not_found:
        new_split = greedy_backward_split(serial,big_text)
        new_splits.append(new_split)

    new_splits = flatten(new_splits)
    all_hits, found, not_found = find_new_splits_in_tables(all_hits,found, not_found, new_splits)

    all_hits, found, not_found = find_empty_and_placeholder_columns({p.casefold(): p for p in parts}, all_hits, found, total_col_count, not_found)
    all_hits, found, not_found = strip_trailing_then_search(all_hits,found, not_found)

    return found, all_hits, not_found


def find_substring_matches_in_tables(tables: List, not_found_parts: List[str]):
    hits = []
    found_parts = []

    # Normalize not_found_parts for case-insensitive matching
    not_found_norm = {part.casefold(): part for part in not_found_parts}

    for table in tables:
        for col_idx, col in enumerate(table.columns):
            for word_idx, word in enumerate(col.texts):
                word_lower = word.lower()
                word_bbox = col.boxes[word_idx]
                word_lower = re.sub(r'[^a-zA-Z0-9_=/ -]', '', word_lower)
                word_lower = word_lower.lstrip()
                # print(word_lower)
                # Check each not-found part
                for norm_part, original_part in not_found_norm.items():
                    if norm_part in word_lower and word_lower.startswith(norm_part):
                        # Avoid double-matching same part
                        # print(word_lower)
                        if original_part in found_parts:
                            continue

                        hits.append({
                            'tableclass': table,
                            "page_index": table.page_index,
                            "table_bbox": table.table_bbox,
                            "column_index": col_idx,
                            "column_bbox": getattr(col, "bbox", None),
                            "matched_text": word,  # full cell text
                            "term": original_part,  # the part we were looking for
                            "word_bbox": word_bbox,
                        })
                        found_parts.append(original_part)

    # Remove found parts from not_found
    remaining_not_found = [p for p in not_found_parts if p not in found_parts]
    return hits, found_parts, remaining_not_found



def greedy_backward_split(serial: str, ocr_text: str):
    """
    Split serial into segments that appear as FULL WORDS in OCR text.

    Matching rules:
    - Only exact full-word matches count (e.g., 'G' matches only if OCR contains 'G' as a word)

    Fallback strategy:
    1. Greedily match longest prefix that is a full OCR word.
    2. If no prefix matches, try splitting remainder into single characters —
       but ONLY if every character exists as a full word in OCR.
    3. If even that fails, append the entire unmatched substring as-is.
    """
    # Normalize OCR text
    ocr_upper = ocr_text.upper()

    # Split key=value patterns into separate tokens (e.g., "RA=REGRESSED" → "RA REGRESSED")
    ocr_processed = re.sub(r"([A-Z0-9/\-]+)=([A-Z0-9/\-]+)", r"\1 \2", ocr_upper)

    # Remove disallowed characters, replace with space
    cleaned_ocr = re.sub(r"[^A-Z0-9/\-\s]", " ", ocr_processed)

    # Normalize whitespace to avoid empty tokens
    cleaned_ocr = ' '.join(cleaned_ocr.split())

    # Extract full words only
    ocr_words = set(word for word in cleaned_ocr.split() if word)

    # Clean serial (keep only allowed chars)
    serial_clean = re.sub(r"[^A-Z0-9/\-]", "", serial.upper().strip())
    segments = []

    def split_recursive(s: str):
        if not s:
            return

        # ✅ Try longest prefix that is a FULL WORD in OCR
        for j in range(len(s), 0, -1):
            candidate = s[:j]
            if candidate in ocr_words:
                # print('candidate',candidate)
                # print(ocr_words)
                segments.append(candidate)
                split_recursive(s[j:])
                return

        # Fallback: try single-character split (only if all chars are valid words)
        if len(s) > 1:
            if all(char in ocr_words for char in s):
                segments.extend(list(s))
                return

        # Final fallback: keep as-is
        segments.append(s)

    split_recursive(serial_clean)
    return segments


def find_empty_and_placeholder_columns(
    parts_lookup: Dict[str, str],
    all_hits: List[Dict[str, Any]],
    found: List[str],
    col_count: int,
    not_found: List[str] = None
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Works on page → tables → hits structure.

    Rules:
      - "blank" is always accepted as a placeholder hit.
      - "xx" is only accepted if not_found contains exactly one numeric-only value.
        In that case, that numeric is consumed and marked as found.
      - If no numeric-only part exists, "xx" does NOT count as a hit.
    """
    updated_found = found.copy()
    updated_all_hits = []
    updated_not_found = not_found.copy() if not_found else []

    serial_parts_length = len(found)
    found_count = col_count

    # already complete, nothing to do
    if serial_parts_length == found_count:
        return all_hits, updated_found, updated_not_found

    # check if not_found has exactly one numeric-only entry
    numeric_only = [nf for nf in updated_not_found if nf.isdigit()]
    numeric_to_use = numeric_only[0] if len(numeric_only) == 1 else None

    for page_result in all_hits:
        page_idx = page_result["page_index"]
        updated_tables = []

        for table_result in page_result["tables"]:
            table = table_result["table"]
            table_hits = table_result["hits"].copy()

            if not table or not hasattr(table, "columns"):
                updated_tables.append({"table": table, "hits": table_hits})
                continue

            col_count = len(table.columns) if hasattr(table.columns, "__len__") else 0
            hit_column_indices = [hit["column_index"] for hit in table_hits if "column_index" in hit]

            for col_idx in range(col_count):
                if col_idx not in hit_column_indices:
                    column = table.columns[col_idx]
                    texts = getattr(column, "texts", []) or []
                    boxes = getattr(column, "boxes", [])

                    for word_idx, word in enumerate(texts):
                        if not word:
                            continue

                        word_lower = re.sub(r'[^a-zA-Z0-9=/-]', '', word.casefold())
                        # print(word_lower)
                        # if word_lower.startswith("blank"):
                        if re.match(r'^.?blank.?', word_lower):
                            # always accept "blank"
                            if word not in updated_found:
                                updated_found.append(word)

                            table_hits.append({
                                "page_index": page_idx,
                                "table_bbox": table.table_bbox,
                                "column_index": col_idx,
                                "column_bbox": getattr(column, "bbox", None),
                                "matched_text": word,
                                "term": '',
                                "word_bbox": boxes[word_idx] if word_idx < len(boxes) else None,
                            })

                        elif "xx" in word_lower and numeric_to_use:
                            # accept "xx" only if numeric candidate exists
                            if word not in updated_found:
                                updated_found.append(word)
                            updated_found.append(numeric_to_use)
                            updated_not_found.remove(numeric_to_use)
                            numeric_to_use = None  # consume once

                            table_hits.append({
                                "page_index": page_idx,
                                "table_bbox": table.table_bbox,
                                "column_index": col_idx,
                                "column_bbox": getattr(column, "bbox", None),
                                "matched_text": word,
                                "term": '',
                                "word_bbox": boxes[word_idx] if word_idx < len(boxes) else None,
                            })

            updated_tables.append({"table": table, "hits": table_hits})

        updated_all_hits.append({"page_index": page_idx, "tables": updated_tables})

    return updated_all_hits, updated_found, updated_not_found



def find_combined_parts_in_tables(
    all_hits: List[Dict[str, Any]],
    found: List[str],
    not_found: List[str]
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Attempts to match concatenated parts from not_found inside tables.
    If found, adds to found & all_hits, and removes from not_found (parts + combo).
    """
    updated_found = found.copy()
    updated_not_found = not_found.copy()
    updated_all_hits = []

    # Generate all possible concatenations (2+ parts)
    combos = []
    for i in range(len(not_found)):
        for j in range(i + 2, len(not_found) + 1):
            parts = not_found[i:j]
            combo_hyphen = "-".join(parts)
            combo_slash = "/".join(parts)
            combos.append((combo_hyphen, parts))
            combos.append((combo_slash, parts))

    # Search inside all pages/tables/columns
    for page_result in all_hits:
        page_idx = page_result["page_index"]
        updated_tables = []

        for table_result in page_result["tables"]:
            table = table_result["table"]
            table_hits = table_result["hits"].copy()

            if not table or not hasattr(table, 'columns'):
                updated_tables.append({"table": table, "hits": table_hits})
                continue

            for col_idx, col in enumerate(table.columns):
                texts = getattr(col, "texts", []) or []
                boxes = getattr(col, "boxes", [])

                for word_idx, word in enumerate(texts):
                    cleaned = re.sub(r'[^a-zA-Z0-9=/-]', '', word).casefold()

                    for combo, parts in combos:
                        if combo.casefold() in cleaned and cleaned.startswith(combo.casefold()):
                            if combo not in updated_found:
                                updated_found.append(combo)

                                # remove matched parts from not_found
                                for p in parts:
                                    if p in updated_not_found:
                                        updated_not_found.remove(p)

                                # also remove the full combo if present
                                if combo in updated_not_found:
                                    updated_not_found.remove(combo)

                            table_hits.append({
                                "page_index": page_idx,
                                "table_bbox": table.table_bbox,
                                "column_index": col_idx,
                                "column_bbox": getattr(col, "bbox", None),
                                "matched_text": word,
                                "term": combo,
                                "word_bbox": boxes[word_idx] if word_idx < len(boxes) else None,
                            })

            updated_tables.append({"table": table, "hits": table_hits})

        updated_all_hits.append({"page_index": page_idx, "tables": updated_tables})

    return updated_all_hits, updated_found, updated_not_found


def strip_trailing_then_search(
    all_hits: List[Dict[str, Any]],
    found: List[str],
    not_found: List[str]
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Strips trailing digits from not_found parts and OCR text,
    then attempts matches inside tables.
    If found, adds to found & all_hits, and removes from not_found.
    """
    updated_found = found.copy()
    updated_not_found = not_found.copy()
    updated_all_hits = []

    def strip_trailing_digits(s: str) -> str:
        return re.sub(r"\d+$", "", s)

    # Prepare stripped versions of not_found
    stripped_map = {nf: strip_trailing_digits(nf) for nf in not_found}

    for page_result in all_hits:
        page_idx = page_result["page_index"]
        updated_tables = []

        for table_result in page_result["tables"]:
            table = table_result["table"]
            table_hits = table_result["hits"].copy()

            if not table or not hasattr(table, "columns"):
                updated_tables.append({"table": table, "hits": table_hits})
                continue

            for col_idx, col in enumerate(table.columns):
                texts = getattr(col, "texts", []) or []
                boxes = getattr(col, "boxes", [])

                for word_idx, word in enumerate(texts):
                    if not word:
                        continue

                    cleaned = re.sub(r"[^a-zA-Z0-9=/ -]", "", word)
                    cleaned_stripped = strip_trailing_digits(cleaned).casefold()
                    cleaned_stripped = cleaned_stripped.lstrip()
                    # compare against stripped not_found values
                    # Extract the first word from the cleaned OCR text
                    first_word = cleaned_stripped.split()[0] if cleaned_stripped.split() else ""

                    # Compare against stripped not_found values
                    for original_nf, stripped_nf in stripped_map.items():
                        stripped_nf_lower = stripped_nf.casefold().strip()
                        if not stripped_nf_lower:
                            continue

                        # Check if the FIRST WORD of OCR text starts with the stripped search term
                        if first_word.startswith(stripped_nf_lower):
                            if original_nf not in updated_found:
                                updated_found.append(original_nf)
                                if original_nf in updated_not_found:
                                    updated_not_found.remove(original_nf)

                            table_hits.append({
                                "page_index": page_idx,
                                "table_bbox": table.table_bbox,
                                "column_index": col_idx,
                                "column_bbox": getattr(col, "bbox", None),
                                "matched_text": word,
                                "term": original_nf,
                                "word_bbox": boxes[word_idx] if word_idx < len(boxes) else None,
                            })

            updated_tables.append({"table": table, "hits": table_hits})

        updated_all_hits.append({"page_index": page_idx, "tables": updated_tables})
    # print('remaining_not_found',updated_not_found)

    return updated_all_hits, updated_found, updated_not_found


def find_reconstructable_codes(found_parts: list[str], original_codes: list[str]) -> list[str]:
    word_set = set(part.strip().casefold() for part in found_parts)
    result = []
    for code in original_codes:
        s = code.strip().casefold()
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        if dp[n]:
            result.append(code)
    return result

def find_new_splits_in_tables(
    all_hits: List[Dict[str, Any]],
    found: List[str],
    not_found: List[str],
    new_splits: List[str]
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Searches directly for new_splits inside tables.
    If found, adds to found & all_hits, removes from not_found.
    """

    updated_found = found.copy()
    updated_not_found = not_found.copy()
    updated_all_hits = []
    # print('new_splits',new_splits)
    combos = [(split, [split]) for split in new_splits]

    for page_result in all_hits:
        found_splits = []
        page_idx = page_result["page_index"]
        updated_tables = []

        for table_result in page_result["tables"]:
            table = table_result["table"]
            table_hits = table_result["hits"].copy()

            if not table or not hasattr(table, 'columns'):
                updated_tables.append({"table": table, "hits": table_hits})
                continue

            for col_idx, col in enumerate(table.columns):
                texts = getattr(col, "texts", []) or []
                boxes = getattr(col, "boxes", [])

                word_idx = 0
                while word_idx < len(texts):
                    word = texts[word_idx]
                    cleaned = re.sub(r'[^a-zA-Z0-9=/() -]', '', word).casefold()

                    for combo, parts in combos[:]:
                        # print(combo,parts)
                        # pattern = r'\b' + re.escape(combo.casefold()) + r'\b'
                        pattern = r'^' + re.escape(combo.casefold()) + r'\b'
                        cleaned = cleaned.lstrip()

                        if re.search(pattern, cleaned):
                            print(pattern, cleaned)
                            if combo not in updated_found:

                                updated_found.append(combo)
                                for p in parts:
                                    found_splits.append(p)
                                    if p in updated_not_found:
                                        updated_not_found.remove(p)

                            table_hits.append({
                                "page_index": page_idx,
                                "table_bbox": table.table_bbox,
                                "column_index": col_idx,
                                "column_bbox": getattr(col, "bbox", None),
                                "matched_text": word,
                                "term": combo,
                                "word_bbox": boxes[word_idx] if word_idx < len(boxes) else None,
                            })

                            combos.remove((combo, parts))

                    word_idx += 1

            updated_tables.append({"table": table, "hits": table_hits})

        updated_all_hits.append({"page_index": page_idx, "tables": updated_tables})
        result = find_reconstructable_codes(found_splits,updated_not_found)
        if result:
          # Find common items
          common_items = set(updated_not_found) & set(result)
          # Remove common items from both lists
          updated_not_found = [item for item in updated_not_found if item not in result]

    return updated_all_hits, updated_found, updated_not_found
