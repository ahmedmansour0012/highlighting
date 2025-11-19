import re
from typing import List

def _split_serial(serial_no: str) -> List[str]:
    # Split on multiple delimiters including spaces
    parts = re.split(r'[-/._:|{}()\[\]\\\s]+', serial_no)
    # Strip whitespace and filter out empty strings
    return [p.strip() for p in parts if p.strip()]

def parse_serial_components(serial_num):
    """
    Parse serial number into individual components.
    Handles various delimiters intelligently.
    """
    # Common delimiters in serial numbers
    delimiters = ['-', '_', '.', '/', '\\', ' ', ',']

    # Start with the original serial number
    components = serial_num

    # Split by each delimiter
    for delimiter in delimiters:
        new_components = []
        for comp in components:
            if delimiter in comp:
                # Split and keep non-empty parts
                parts = [p.strip() for p in comp.split(delimiter) if p.strip()]
                new_components.extend(parts)
            else:
                new_components.append(comp)
        components = new_components

    # Remove duplicates while preserving order
    seen = set()
    unique_components = []
    for comp in components:
        if comp not in seen:
            seen.add(comp)
            unique_components.append(comp)

    return unique_components


def filter_components(components):
    """
    Filter components to reduce false positives.
    Removes single characters and very common patterns.
    """
    filtered = []

    for comp in components:
        # Skip single characters (too many false positives)
        if len(comp) <= 1:
            continue

        # Skip pure numbers less than 3 digits (too common)
        if comp.isdigit() and len(comp) < 3:
            continue

        # Keep components that are:
        # - Alphanumeric with length >= 3
        # - Mixed alphanumeric (letters and numbers)
        # - Pure letters with length >= 3
        # - Pure numbers with length >= 3

        has_letter = any(c.isalpha() for c in comp)
        has_digit = any(c.isdigit() for c in comp)

        # Prioritize mixed alphanumeric (most distinctive)
        if has_letter and has_digit:
            filtered.append(comp)
        # Accept longer pure letter or number sequences
        elif len(comp) >= 3:
            filtered.append(comp)
        # Accept 2-character sequences if they're not too common
        elif len(comp) == 2 and comp.upper():
            # You might want to be more selective here
            filtered.append(comp)

    return filtered
