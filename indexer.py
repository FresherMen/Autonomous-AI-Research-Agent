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

    def __init__(self):
        self.chunker = DocumentChunker(chunk_size=500, overlap=50)
        self.embedder = Embedder()                    # loads model once
        self.store    = VectorStore("./chroma_db")    # persists to disk

    def index_file(self, file_path: str):
        print(f"\nIndexing: {file_path}")

        # Step 1: Chunk
        if file_path.endswith(".pdf"):
            chunks = self.chunker.chunk_pdf(file_path)
        else:
            chunks = self.chunker.chunk_file(file_path)

        if not chunks:
            print("  No content found. Skipping.")
            return

        # Step 2: Embed all chunks in one batch (much faster than one at a time)
        print(f"  Embedding {len(chunks)} chunks...")
        texts      = [c['text'] for c in chunks]
        embeddings = self.embedder.embed(texts)

        # Step 3: Store in ChromaDB
        self.store.add_chunks(chunks, embeddings)
        print(f"  Done. Index now has {self.store.count()} total chunks.")

    def index_folder(self, folder_path: str):
        """Index every .pdf, .txt, and .md file in a folder."""
        supported = ('.pdf', '.txt', '.md')
        files = [f for f in os.listdir(folder_path) if f.endswith(supported)]
        print(f"Found {len(files)} documents in '{folder_path}'")
        for filename in files:
            self.index_file(os.path.join(folder_path, filename))

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """
        Search the index with a plain English query.
        Used in Step 3 to retrieve context before answering.
        """
        query_vector = self.embedder.embed_one(query)
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