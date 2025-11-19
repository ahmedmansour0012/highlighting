import re

from .parsing import filter_components, parse_serial_components, _split_serial


# Configure logging once (optional)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)  # Change to DEBUG to see detailed trace

# Common constants
KEYWORDS = [
    "order", "ordering", "logic", "guide",
    "product selector", "information", "specification"
]


# ----------------------------
# Helper utilities
# ----------------------------

def preprocess_ocr_texts(ocr_results):
    """Precompute normalized lowercase text for each page."""
    processed_pages = []
    for idx, page_result in enumerate(ocr_results):
        texts = page_result[0].get('rec_texts', [])
        joined_text = " ".join(texts).lower()
        processed_pages.append(joined_text)
        print(f"Page {idx}: preprocessed text length={len(joined_text)}")
    return processed_pages


def compile_keyword_pattern(keywords):
    """Compile a single regex pattern for all keywords (case-insensitive)."""
    escaped = [re.escape(k) for k in keywords]
    return re.compile(r"(?:{})".format("|".join(escaped)), re.IGNORECASE)


def search_by_keywords(pages_text, keywords):
    """Return list of page indices where any keyword appears."""
    pattern = compile_keyword_pattern(keywords)
    matches = [idx for idx, text in enumerate(pages_text) if pattern.search(text)]
    print(f"Keyword search found {len(matches)} pages: {matches}")
    return matches


import re

def match_serial_components(page_text, components, threshold_ratio=0.7):
    """
    Count matched components and return confidence score.
    Ensures exact (whole word) matches — not substrings.
    Returns (confidence, matched_components)
    """
    total = len(components)
    if total == 0:
        return 0.0, []

    # Split page text into normalized tokens for exact matching
    tokens = re.findall(r"[A-Za-z0-9\-_/\.]+", page_text.lower())
    token_set = set(tokens)  # faster lookups
    matched_components = []

    for comp in components:
        comp_lower = comp.lower()
        if comp_lower in token_set:
            matched_components.append(comp)

    matches = len(matched_components)

    # Adaptive threshold logic
    if total <= 3:
        threshold = total
    elif total <= 5:
        threshold = total - 1
    else:
        threshold = int(total * threshold_ratio)

    passed = matches >= threshold
    confidence = matches / total if passed else 0.0

    print(
        f"Components matched: {matched_components}, "
        f"{matches}/{total}, threshold={threshold}, confidence={confidence:.2f}"
    )
    return confidence, matched_components


# ----------------------------
# Main logic
# ----------------------------

# def find_pages(ocr_results, serial_num):
#     """
#     Find pages likely related to the serial number.
#     - If multiple full serial matches found, filter by component match.
#     - Falls back to component or keyword search.
#     """
#     pages_text = preprocess_ocr_texts(ocr_results)
#     separator = "-"
#     full_compined = separator.join(str(element) for element in serial_num)
#     serial_lower = full_compined.lower()
#     pages = []

#     # 1️⃣ Full serial match
#     full_matches = [idx for idx, text in enumerate(pages_text) if serial_lower in text]
#     print(f"Full serial matches: {full_matches}")

#     # 2️⃣ If multiple full matches, filter them by component confirmation
#     if len(full_matches) > 1:
#         components = filter_components(parse_serial_components(serial_num))
#         print(f"Filtered components: {components}")
#         confirmed = []

#         for idx in full_matches:
#             conf, matched = match_serial_components(pages_text[idx], components)
#             if conf > 0:
#                 confirmed.append(idx)
#                 print(f"Page {idx} confirmed by component match (conf={conf:.2f})")

#         # keep only confirmed matches if any were found
#         pages = confirmed if confirmed else full_matches
#         print(f"Filtered full matches after component check: {pages}")

#     else:
#         pages = full_matches

#     # 3️⃣ If still no pages, fall back to component-based search across all pages
#     if not pages:
#         components = filter_components(parse_serial_components(serial_num))
#         print(f"Filtered components: {components}")
#         for idx, text in enumerate(pages_text):
#             conf, matched = match_serial_components(text, components)
#             if conf > 0:
#                 pages.append(idx)
#                 print(f"Page {idx} passed component match (conf={conf:.2f})")

#     # 4️⃣ Keyword fallback
#     if not pages:
#         pages = search_by_keywords(pages_text, KEYWORDS)
#         print(f"Keyword fallback matches: {pages}")

#     print(f"Final matched pages: {pages}")
#     return pages

def find_pages(ocr_results, serial_num):
    """
    Find pages likely related to the serial number.
    - If multiple full serial matches found, filter by component match.
    - Falls back to component or keyword search.
    Returns tuple: (pages, matched_components_dict)
    """
    pages_text = preprocess_ocr_texts(ocr_results)
    separator = "-"
    full_compined = separator.join(str(element) for element in serial_num)
    serial_lower = full_compined.lower()
    pages = []
    matched_components_dict = {}  # Dictionary to store matched components per page

    # 1️⃣ Full serial match
    full_matches = [idx for idx, text in enumerate(pages_text) if serial_lower in text]
    print(f"Full serial matches: {full_matches}")

    # 2️⃣ If multiple full matches, filter them by component confirmation
    if len(full_matches) > 1:
        components = filter_components(parse_serial_components(serial_num))
        print(f"Filtered components: {components}")
        confirmed = []

        for idx in full_matches:
            conf, matched = match_serial_components(pages_text[idx], components)
            if conf > 0:
                confirmed.append(idx)
                matched_components_dict[idx] = {
                    'confidence': conf,
                    'matched_components': matched,
                    'match_type': 'full_match_with_component'
                }
                print(f"Page {idx} confirmed by component match (conf={conf:.2f}, components={matched})")

        # keep only confirmed matches if any were found
        pages = confirmed if confirmed else full_matches
        print(f"Filtered full matches after component check: {pages}")

    else:
        # Handle single full match
        if full_matches:
            pages = full_matches
            # Still check for component matches even with single full match
            components = filter_components(parse_serial_components(serial_num))
            for idx in full_matches:
                conf, matched = match_serial_components(pages_text[idx], components)
                if conf > 0:
                    matched_components_dict[idx] = {
                        'confidence': conf,
                        'matched_components': matched,
                        'match_type': 'full_match_with_component'
                    }
                else:
                    matched_components_dict[idx] = {
                        'confidence': 1.0,  # Full match confidence
                        'matched_components': [],
                        'match_type': 'full_match_only'
                    }

    # 3️⃣ If still no pages, fall back to component-based search across all pages
    if not pages:
        components = filter_components(parse_serial_components(serial_num))
        print(f"Filtered components: {components}")
        for idx, text in enumerate(pages_text):
            conf, matched = match_serial_components(text, components)
            if conf > 0:
                pages.append(idx)
                matched_components_dict[idx] = {
                    'confidence': conf,
                    'matched_components': matched,
                    'match_type': 'component_match_only'
                }
                print(f"Page {idx} passed component match (conf={conf:.2f}, components={matched})")

    # 4️⃣ Keyword fallback
    if not pages:
        keyword_pages = search_by_keywords(pages_text, KEYWORDS)
        print(f"Keyword fallback matches: {keyword_pages}")
        pages = keyword_pages
        for idx in keyword_pages:
            # Even for keyword matches, check if any components were found
            components = filter_components(parse_serial_components(serial_num))
            conf, matched = match_serial_components(pages_text[idx], components)
            matched_components_dict[idx] = {
                'confidence': conf if conf > 0 else 0.0,
                'matched_components': matched if conf > 0 else [],
                'match_type': 'keyword_match' + ('_with_components' if conf > 0 else '')
            }

    print(f"Final matched pages: {pages}")
    print(f"Matched components per page: {matched_components_dict}")
    return pages, matched_components_dict

def find_pages_with_confidence(ocr_results, serial_num):
    """
    Return pages with confidence scores for full, component, or keyword matches.
    """
    pages_text = preprocess_ocr_texts(ocr_results)
    serial_lower = serial_num.lower()
    results = []

    # 1️⃣ Full serial match
    for idx, text in enumerate(pages_text):
        if serial_lower in text:
            results.append({
                'page': idx,
                'confidence': 1.0,
                'match_type': 'full'
            })
    print(f"Full serial match results: {results}")

    if results:
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    # 2️⃣ Component matching
    components = filter_components(parse_serial_components(serial_num))
    print(f"Filtered components: {components}")

    for idx, text in enumerate(pages_text):
        confidence, matched = match_serial_components(text, components)
        if confidence > 0:
            results.append({
                'page': idx,
                'confidence': round(confidence, 2),
                'match_type': 'components',
                'matched_components': matched
            })
            print(f"Page {idx} component conf={confidence:.2f}, matched={matched}")

    # 3️⃣ Keyword fallback (low confidence)
    if not results:
        for idx, text in enumerate(pages_text):
            keyword_matches = [k for k in KEYWORDS if k in text]
            if keyword_matches:
                conf = 0.3 * (len(keyword_matches) / len(KEYWORDS))
                results.append({
                    'page': idx,
                    'confidence': round(conf, 2),
                    'match_type': 'keywords',
                    'matched_keywords': keyword_matches
                })
                print(f"Page {idx} keyword fallback conf={conf:.2f}, matched={keyword_matches}")

    results = sorted(results, key=lambda x: x['confidence'], reverse=True)
    print(f"Final results: {results}")
    return results

def match_serials_to_pages(serial_numbers, all_matched_components_dict):
    """
    Match each serial number to its best page based on component overlap.
    
    Args:
        serial_numbers: List of serial numbers to match
        all_matched_components_dict: Dict with page -> {confidence, matched_components, match_type}
    
    Returns:
        Dict mapping each serial number to its matched page number(s)
    """
    serial_to_page = {}
    
    for serial_num in serial_numbers:
        # Parse components from the current serial number
        serial_components = _split_serial(serial_num)
        
        best_page = None
        best_overlap_score = 0
        
        # Check each page for overlap
        for page_num, page_info in all_matched_components_dict.items():
            page_matched_components = page_info['matched_components']
            
            # Find overlap between serial components and page components
            overlapping_components = set(serial_components) & set(page_matched_components)
            print(overlapping_components,serial_components,page_matched_components)
            if overlapping_components:
                overlap_score = len(overlapping_components) / len(serial_components)
                
                # Update best match if this one is better
                print(overlap_score)
                if overlap_score > best_overlap_score:
                    best_overlap_score = overlap_score
                    best_page = page_num
        
        # Store the best matching page for this serial number
        # If no match found, store None or empty list based on your preference
        serial_to_page[serial_num] = best_page if best_page is not None else []
    
    return serial_to_page