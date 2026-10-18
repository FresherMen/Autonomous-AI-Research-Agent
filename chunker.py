import re

class DocumentChunker:
    """
    Splits raw text into overlapping chunks.
    Think of it like cutting a long scroll into
    overlapping index cards.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source_name: str) -> list[dict]:
        """
        Returns a list of chunk dicts, each with:
          - text:   the passage itself
          - source: which file it came from
          - index:  which chunk number (for debugging)
        """
        # Clean up extra whitespace first
        text = re.sub(r'\s+', ' ', text).strip()

        chunks = []
        step = self.chunk_size - self.overlap   # how far we advance each time
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to end at a sentence boundary instead of mid-word
            if end < len(text):
                last_period = chunk_text.rfind('.')
                if last_period > self.chunk_size * 0.5:  # only if period is in the second half
                    chunk_text = chunk_text[:last_period + 1]

            chunks.append({
                "text":   chunk_text,
                "source": source_name,
                "index":  len(chunks)
            })

            start += step

        print(f"  '{source_name}' → {len(chunks)} chunks")
        return chunks

    def chunk_pdf(self, pdf_path: str) -> list[dict]:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        source_name = pdf_path.split("/")[-1]   # just the filename
        return self.chunk_text(full_text, source_name)

    def chunk_file(self, file_path: str) -> list[dict]:
        """Load a .txt or .md file and chunk it."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        source_name = file_path.split("/")[-1]
        return self.chunk_text(text, source_name)