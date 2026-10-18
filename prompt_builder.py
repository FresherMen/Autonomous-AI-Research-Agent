class PromptBuilder:
    """
    Turns a question + a list of retrieved chunks
    into a complete, structured prompt for the LLM.
    
    This is a simple class but it's worth isolating —
    you'll iterate on the template a lot as you improve
    your assistant's answer quality.
    """

    def __init__(self, max_context_chars: int = 3000):
        # Safety limit: don't let context balloon the token count
        self.max_context_chars = max_context_chars

    def build(self, question: str, chunks: list[dict]) -> str:
        """
        Args:
            question: the user's raw question
            chunks:   list of dicts from VectorStore.search()
                      each has: text, source, similarity

        Returns:
            A complete prompt string ready to send to the LLM
        """
        context_block = self._build_context_block(chunks)

        prompt = f"""You are a research assistant. Answer the question using 
ONLY the context provided below. Cite which source each piece of 
information came from. If the context doesn't contain enough information 
to answer, say exactly that — do not guess or use outside knowledge.

CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:"""
        return prompt

    def _build_context_block(self, chunks: list[dict]) -> str:
        """Format each chunk with its source label."""
        lines = []
        total_chars = 0

        for i, chunk in enumerate(chunks):
            header  = f"[Source: {chunk['source']} | Similarity: {chunk['similarity']}]"
            passage = chunk['text'].strip()
            entry   = f"{header}\n{passage}\n"

            # Stop adding chunks if we'd exceed the character budget
            if total_chars + len(entry) > self.max_context_chars:
                lines.append(f"[... {len(chunks) - i} more chunks omitted for length]")
                break

            lines.append(entry)
            total_chars += len(entry)

        return "\n".join(lines)

    def build_no_results_prompt(self, question: str) -> str:
        """Used when the vector search returns nothing relevant."""
        return f"""You are a research assistant. The user asked a question,
but no relevant documents were found in the knowledge base.

Politely inform them that their document collection doesn't seem to contain 
information about this topic. Suggest they either:
1. Upload more relevant documents
2. Rephrase their question
3. Ask about something else

QUESTION: {question}

ANSWER:"""