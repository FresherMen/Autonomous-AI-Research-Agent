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
        if not chunks or not embeddings:
            return

        # Ensure lengths match
        min_len = min(len(chunks), len(embeddings))
        chunks = chunks[:min_len]
        embeddings = embeddings[:min_len]

        ids         = [f"{c['source']}::chunk{c.get('index', i)}" for i, c in enumerate(chunks)]
        documents   = [c.get('text', '') for c in chunks]
        metadatas   = [{"source": c.get('source', 'unknown'), "index": c.get('index', i)} for i, c in enumerate(chunks)]

        # Deduplicate — don't add the same chunk twice if we re-run
        try:
            get_res = self.collection.get(ids=ids)
            existing_ids = set(get_res.get('ids', [])) if get_res else set()
        except Exception:
            existing_ids = set()

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
        total_count = self.collection.count()
        if total_count == 0 or top_k <= 0:
            return []

        n_results = min(top_k, total_count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        docs = results.get('documents')
        metas = results.get('metadatas')
        dists = results.get('distances')

        if not docs or len(docs) == 0 or not docs[0]:
            return []

        hits = []
        meta_list = metas[0] if (metas and len(metas) > 0) else [{}] * len(docs[0])
        dist_list = dists[0] if (dists and len(dists) > 0) else [None] * len(docs[0])

        for text, meta, dist in zip(docs[0], meta_list, dist_list):
            meta_dict = meta if isinstance(meta, dict) else {}
            hits.append({
                "text":       text,
                "source":     meta_dict.get('source', 'unknown'),
                "chunk_index": meta_dict.get('index', 0),
                "similarity": round(1 - dist, 3) if dist is not None else 0.0
            })

        return hits

    def count(self) -> int:
        return self.collection.count()