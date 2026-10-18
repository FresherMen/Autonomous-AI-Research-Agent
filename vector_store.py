import chromadb
from chromadb.config import Settings

class VectorStore:
    """
    A wrapper around ChromaDB.
    
    ChromaDB stores three things per chunk:
      1. The chunk text itself  (so we can return readable results)
      2. The vector             (so we can do similarity search)
      3. Metadata               (source filename, chunk index)
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        # persist_dir = folder where ChromaDB saves data to disk
        # This means your index survives between runs!
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="research_docs",
            metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
        )
        print(f"VectorStore ready. Contains {self.collection.count()} chunks.")

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        """
        Store chunks + their vectors in ChromaDB.
        
        ChromaDB needs:
          ids:        unique string ID per chunk
          documents:  the raw text
          embeddings: the vector
          metadatas:  any extra info (source file, index)
        """
        ids         = [f"{c['source']}::chunk{c['index']}" for c in chunks]
        documents   = [c['text']   for c in chunks]
        metadatas   = [{"source": c['source'], "index": c['index']} for c in chunks]

        # Deduplicate — don't add the same chunk twice if we re-run
        existing_ids = set(self.collection.get(ids=ids)['ids'])
        new_items = [(i, d, e, m) for i, d, e, m in
                     zip(ids, documents, embeddings, metadatas)
                     if i not in existing_ids]

        if not new_items:
            print("  All chunks already indexed. Skipping.")
            return

        new_ids, new_docs, new_embs, new_metas = zip(*new_items)
        self.collection.add(
            ids=list(new_ids),
            documents=list(new_docs),
            embeddings=list(new_embs),
            metadatas=list(new_metas)
        )
        print(f"  Added {len(new_ids)} new chunks. Total: {self.collection.count()}")

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """
        Find the top_k most similar chunks to a query vector.
        Returns a list of dicts with text, source, and similarity score.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for text, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            hits.append({
                "text":       text,
                "source":     meta['source'],
                "chunk_index": meta['index'],
                "similarity": round(1 - dist, 3)  # ChromaDB returns distance; flip to similarity
            })

        return hits

    def count(self) -> int:
        return self.collection.count()