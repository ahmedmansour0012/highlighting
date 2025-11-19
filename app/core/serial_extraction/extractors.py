import re
from serialnumbers import SerialNumber

def find_classic_serials(text):
    """
    Extract classic serial numbers / catalog codes like:
    PS51974822, 4CZL-9-LVR-UNV-L840-WLS4-CD-U, CIRRUS-SN
    Context-aware and line-based.
    """
    context_phrases = [
        "example:", "part number:", "figure:", "serial no:",
        "ordering code:", "order code:", "ex:", "ordering logic   ex"
    ]
    context_patterns = [re.compile(r"sample\s+\w+\s+number", re.I)]

    results = []
    lines = text.splitlines()

    for i, line in enumerate(lines):
        lower_line = line.lower()
        if any(p.search(lower_line) for p in context_patterns) or \
           any(phrase.lower() in lower_line for phrase in context_phrases):

            candidate_text = line
            if i + 1 < len(lines):
                candidate_text += " " + lines[i + 1]

            # for match in re.finditer(r"[A-Za-z0-9\-]{3,}", candidate_text):
            #     token = match.group()

            #     # allow codes with digits OR at least 2 segments separated by '-'
            #     if not (re.search(r"\d", token) or "-" in token):
            #         continue
            #     if len(token) < 4:
            #         continue

            #     try:
            #         sn = SerialNumber.fromString(token)
            #         results.append(sn.toString())
            #     except ValueError:
            #         # If SerialNumber rejects it, keep it if it looks like a hyphenated code
            #         if "-" in token and token.isalnum() is False:
            #             results.append(token)
            # Change the regex to allow hyphens AND slashes
            for match in re.finditer(r"[A-Za-z0-9\-/]{3,}", candidate_text):
                token = match.group()

                # Now accept tokens with digits, hyphens, OR slashes
                if not (re.search(r"\d", token) or "-" in token or "/" in token):
                    continue
                if len(token) < 4:
                    continue

                try:
                    sn = SerialNumber.fromString(token)
                    results.append(sn.toString())
                except ValueError:
                    # Accept if it has - or /
                    if ("-" in token or "/" in token) and not token.isalnum():
                        results.append(token)
    return results


def find_space_separated_codes(text, min_segments=3):
    """
    Extract space-separated product codes (e.g. 'GSLF3 P30 40K MVOLT ASY QSM BK')
    line-aware, requires context phrase nearby.
    """
    context_patterns = [
        re.compile(r"sample\s+\w+\s+number", re.I),
        re.compile(r"example:", re.I),
    ]

    results = []
    lines = text.splitlines()
    space_code_pattern = re.compile(r"(?:[A-Z0-9]{2,}\s+){2,}[A-Z0-9]{2,}", re.I)

    for i, line in enumerate(lines):
        lower_line = line.lower()

        # If line has context phrase
        if any(p.search(lower_line) for p in context_patterns):

            # keep only the text AFTER the context keyword
            for pat in context_patterns:
                m = pat.search(lower_line)
                if m:
                    start = m.end()
                    candidate_text = line[start:].strip()
                    break
            else:
                candidate_text = line

            # also check next line in case code is split
            if i + 1 < len(lines):
                candidate_text += " " + lines[i + 1]

            # now run regex only on candidate portion
            for match in space_code_pattern.finditer(candidate_text):
                code = " ".join(match.group().split())
                if len(code.split()) >= min_segments and "number" not in code.lower():
                    results.append(code)

    return results
