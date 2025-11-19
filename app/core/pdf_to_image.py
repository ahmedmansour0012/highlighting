import fitz  # PyMuPDF
from PIL import Image

def convert_pdf_with_pymupdf(pdf_path=None, pdf_bytes=None):
    """
    Convert a PDF (from file path or bytes) into a list of PIL Image objects.

    Args:
        pdf_path (str, optional): Path to the PDF file.
        pdf_bytes (bytes, optional): Raw PDF data in bytes.

    Returns:
        list[PIL.Image.Image]: List of images, one per page.
    """
    if not pdf_path and not pdf_bytes:
        raise ValueError("You must provide either 'pdf_path' or 'pdf_bytes'.")

    # Open the document appropriately
    if pdf_bytes:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    else:
        doc = fitz.open(pdf_path)

    images = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pixmap = page.get_pixmap(dpi=180)

        # Convert pixmap to PIL Image
        mode = "RGB"
        img = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
        images.append(img)

    doc.close()
    return images
