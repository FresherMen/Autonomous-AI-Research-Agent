import os
import re

class DocumentChunker:
    """
    Splits raw text into overlapping chunks.
    Think of it like cutting a long scroll into
    overlapping index cards.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = max(10, int(chunk_size))
        self.overlap = max(0, min(int(overlap), self.chunk_size - 1))

    def chunk_text(self, text: str, source_name: str = "unknown", **kwargs) -> list[dict]:
        """
        Returns a list of chunk dicts, each with:
          - text:   the passage itself
          - source: which file it came from
          - index:  which chunk number (for debugging)
        """
        source_name = kwargs.get("source", source_name)
        # Clean up extra whitespace first
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to end at a sentence boundary instead of mid-word
            if end < len(text):
                half = int(len(chunk_text) * 0.5)
                best_break = -1
                for punct in ('. ', '? ', '! ', '.\n', '?\n', '!\n'):
                    pos = chunk_text.rfind(punct, half)
                    if pos != -1 and (best_break == -1 or pos + 1 > best_break):
                        best_break = pos + 1
                if best_break == -1:
                    pos = chunk_text.rfind('.', half)
                    if pos != -1:
                        best_break = pos + 1

                if best_break > half:
                    chunk_text = chunk_text[:best_break]

            cleaned = chunk_text.strip()
            if cleaned:
                chunks.append({
                    "text":   cleaned,
                    "source": str(source_name),
                    "index":  len(chunks)
                })

            if end >= len(text):
                break

            # Advance without skipping intermediate text
            next_start = start + len(chunk_text) - self.overlap
            start = max(start + 1, next_start)

        print(f"  '{source_name}' → {len(chunks)} chunks")
        return chunks

    def chunk_pdf(self, pdf_path: str) -> list[dict]:
        import pypdf
        if not os.path.exists(pdf_path):
            print(f"  File not found: {pdf_path}")
            return []

        full_text = ""
        try:
            reader = pypdf.PdfReader(pdf_path)
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                except Exception as pe:
                    print(f"  Warning: failed to extract text from a page in {pdf_path}: {pe}")
        except Exception as e:
            print(f"  Error reading PDF '{pdf_path}': {e}")
            return []

        source_name = os.path.basename(pdf_path)
        return self.chunk_text(full_text, source_name)

    def chunk_file(self, file_path: str) -> list[dict]:
        """Load a .txt or .md file and chunk it."""
        if not os.path.exists(file_path):
            print(f"  File not found: {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            print(f"  Error reading file '{file_path}': {e}")
            return []

        source_name = os.path.basename(file_path)
        return self.chunk_text(text, source_name)
