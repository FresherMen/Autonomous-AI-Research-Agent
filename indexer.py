import os
from chunker import DocumentChunker
from embedder import Embedder
from vector_store import VectorStore

class DocumentIndexer:
    """
    The full indexing pipeline:
      Load docs → Chunk → Embed → Store
    
    This is the 'one time setup' that runs whenever
    you add new documents to your research assistant.
    """

    def __init__(self, store: VectorStore = None, persist_dir: str = "./chroma_db"):
        self.chunker  = DocumentChunker(chunk_size=500, overlap=50)
        self.embedder = Embedder()                    # loads model once
        self.store    = store or VectorStore(persist_dir)  # persists to disk

    def index_file(self, file_path: str) -> int:
        """
        Index a single file (.pdf, .txt, .md).
        Returns the number of chunks added (0 if skipped or failed).
        """
        print(f"\nIndexing: {file_path}")
        if not os.path.exists(file_path):
            print(f"  File not found: {file_path}. Skipping.")
            return 0

        lower_path = file_path.lower()
        # Step 1: Chunk
        try:
            if lower_path.endswith(".pdf"):
                chunks = self.chunker.chunk_pdf(file_path)
            elif lower_path.endswith((".txt", ".md")):
                chunks = self.chunker.chunk_file(file_path)
            else:
                print(f"  Skipping unsupported file format: {file_path}")
                return 0
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            return 0

        if not chunks:
            print("  No content found. Skipping.")
            return 0

        # Step 2: Embed all chunks in one batch (much faster than one at a time)
        print(f"  Embedding {len(chunks)} chunks...")
        try:
            texts      = [c['text'] for c in chunks]
            embeddings = self.embedder.embed(texts)
        except Exception as e:
            print(f"  Error generating embeddings for {file_path}: {e}")
            return 0

        # Step 3: Store in ChromaDB
        try:
            self.store.add_chunks(chunks, embeddings)
            print(f"  Done. Index now has {self.store.count()} total chunks.")
            return len(chunks)
        except Exception as e:
            print(f"  Error storing chunks for {file_path}: {e}")
            return 0

    def index_folder(self, folder_path: str) -> int:
        """Index every .pdf, .txt, and .md file in a folder. Returns total chunks added."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            print(f"Folder '{folder_path}' created (empty).")
            return 0

        supported = ('.pdf', '.txt', '.md')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported)]
        print(f"Found {len(files)} documents in '{folder_path}'")
        total_indexed = 0
        for filename in files:
            try:
                count = self.index_file(os.path.join(folder_path, filename))
                total_indexed += count
            except Exception as e:
                print(f"  Failed indexing {filename}: {e}")
        return total_indexed

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """
        Search the index with a plain English query.
        Used in Step 3 to retrieve context before answering.
        """
        if not query or not query.strip():
            return []
        query_vector = self.embedder.embed_one(query.strip())
        return self.store.search(query_vector, top_k=top_k)


# ---  Run this file directly to index your documents ---
if __name__ == "__main__":
    indexer = DocumentIndexer()

    # Put your PDFs and notes in a folder called 'docs/'
    indexer.index_folder("./docs")

    # Test a search immediately after indexing
    print("\n--- Test Search ---")
    results = indexer.search("What is dropout regularization?", top_k=3)
    for r in results:
        print(f"\n[{r['source']}] similarity={r['similarity']}")
        print(r['text'][:200] + "...")
