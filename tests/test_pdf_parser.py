from backend.pdf_parser import extract_text_from_pdf

def test_pdf_parser_empty():
    result = extract_text_from_pdf(b"")
    assert isinstance(result, list)
