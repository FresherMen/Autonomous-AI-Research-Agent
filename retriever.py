import os
from dotenv import load_dotenv
from groq import Groq
from indexer import DocumentIndexer
from prompt_builder import PromptBuilder

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class RAGRetriever:

    RELEVANCE_THRESHOLD = 0.35

    def __init__(self):
        self.indexer  = DocumentIndexer()
        self.builder  = PromptBuilder(max_context_chars=3000)
        self.history  = []
        self.system_prompt = (
            "You are a helpful research assistant. "
            "Answer questions based on the documents the user has uploaded. "
            "Always be honest when the documents don't cover something."
        )

    def ask(self, question: str, top_k: int = 4, verbose: bool = False) -> dict:
        chunks = self.indexer.search(question, top_k=top_k)
        relevant_chunks = [c for c in chunks
                           if c['similarity'] >= self.RELEVANCE_THRESHOLD]

        if relevant_chunks:
            augmented_prompt = self.builder.build(question, relevant_chunks)
        else:
            augmented_prompt = self.builder.build_no_results_prompt(question)

        self.history.append({"role": "user", "content": augmented_prompt})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            temperature=0.2,
            messages=[
                {"role": "system", "content": self.system_prompt}
            ] + self.history
        )

        answer = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer":  answer,
            "chunks":  relevant_chunks,
            "tokens":  {
                "input":  response.usage.prompt_tokens,
                "output": response.usage.completion_tokens
            }
        }

    def reset_conversation(self):
        self.history = []
        print("Conversation history cleared.")