from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Wraps a sentence-transformers model.
    
    Why sentence-transformers?
    - Free, runs locally, no API key needed
    - 'all-MiniLM-L6-v2' is tiny (80MB) but surprisingly good
    - Downloads once, cached forever after
    
    In production you'd swap this for OpenAI's
    text-embedding-3-small for better quality.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        print("  Model ready.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of strings → list of vectors.
        Each vector is a list of ~384 floats.
        """
        vectors = self.model.encode(texts, show_progress_bar=True)
        return vectors.tolist()   # ChromaDB wants plain Python lists

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string — used at query time."""
        return self.model.encode([text])[0].tolist()