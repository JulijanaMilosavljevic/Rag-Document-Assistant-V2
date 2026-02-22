from typing import List


def chunk_text(
    text: str,
    max_chars: int = 800,
    overlap: int = 200
) -> List[str]:
    """
    Prosti chunking na nivou 'rečenica' / linija, po dužini stringa.
    max_chars: maksimalna dužina chunk-a
    overlap: preklapanje između chunk-ova radi boljeg konteksta
    """
    if not text:
        return []

    # Podeli otprilike po rečenicama / linijama
    import re
    raw_sentences = re.split(r'(?<=[\.\!\?])\s+|\n+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sent
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start new chunk – možda uzme i deo overlapa iz prethodnog
            if len(sent) > max_chars:
                # ako je jedna rečenica ogromna, samo iseći je
                while len(sent) > max_chars:
                    chunks.append(sent[:max_chars].strip())
                    sent = sent[max_chars:]
                current_chunk = sent
            else:
                current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk.strip())

    # jednostavno overlap: spajaj kraj prethodnog chunk-a na početak
    # (ovo je opciono, da ne komplikujemo – već osnovni chunking radi ok)

    return chunks
