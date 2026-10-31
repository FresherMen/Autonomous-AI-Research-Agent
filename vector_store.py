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

        unique_items = []
        seen_batch_ids = set()
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            src = str(chunk.get("source") or "unknown")
            idx = chunk.get("index") if chunk.get("index") is not None else i
            chunk_id = f"{src}::chunk{idx}"
            if chunk_id in seen_batch_ids:
                chunk_id = f"{chunk_id}_{i}"
            seen_batch_ids.add(chunk_id)

            unique_items.append((
                chunk_id,
                str(chunk.get("text", "")),
                emb,
                {"source": src, "index": idx}
            ))

        ids = [item[0] for item in unique_items]

        # Deduplicate — don't add the same chunk twice if we re-run
        try:
            get_res = self.collection.get(ids=ids)
            existing_ids = set(get_res.get('ids', [])) if get_res else set()
        except Exception:
            existing_ids = set()

        new_items = [item for item in unique_items if item[0] not in existing_ids]

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
        if not query_embedding:
            return []

        try:
            top_k = int(top_k)
        except (ValueError, TypeError):
            top_k = 4

        total_count = self.collection.count()
        if total_count == 0 or top_k <= 0:
            return []

        n_results = min(top_k, total_count)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

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
            if dist is not None:
                sim = round(max(-1.0, min(1.0, 1.0 - dist)), 3)
            else:
                sim = 0.0

            hits.append({
                "text":        text,
                "source":      meta_dict.get('source', 'unknown'),
                "chunk_index": meta_dict.get('index', 0),
                "similarity":  sim
            })

        return hits

    def list_sources(self) -> list[str]:
        """Return a sorted list of unique source document names."""
        try:
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas") or []
            return sorted(set(
                str(m["source"]) for m in metadatas
                if m and isinstance(m, dict) and m.get("source")
            ))
        except Exception as e:
            print(f"Error listing sources: {e}")
            return []

    def count(self) -> int:
        return self.collection.count()
