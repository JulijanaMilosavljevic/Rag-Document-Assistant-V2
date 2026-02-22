from typing import List, Tuple
from pypdf import PdfReader
from io import BytesIO


def extract_text_from_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Vraća listu (page_number, text) za svaki page u PDF-u.
    page_number kreće od 1 radi lepšeg prikaza.
    """
    pages = []
    pdf_stream = BytesIO(file_bytes)
    reader = PdfReader(pdf_stream)

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.replace("\u00a0", " ")  # non-breaking space
        pages.append((i + 1, text.strip()))

    return pages
