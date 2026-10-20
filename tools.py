import os
import json
import datetime
from retriever import RAGRetriever     # Step 3 — our RAG pipeline

# ─────────────────────────────────────────────
#  Tool schemas — given to the LLM so it knows
#  what tools exist and what arguments they take
# ─────────────────────────────────────────────

# TOOL_SCHEMAS = [
#     {
#         "name": "search_docs",
#         "description": (
#             "Search the user's uploaded document collection. "
#             "Use this when the question is about something the user "
#             "may have in their research papers, notes, or PDFs."
#         ),
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "A clear search query to find relevant passages"
#                 },
#                 "top_k": {
#                     "type": "integer",
#                     "description": "How many chunks to retrieve (default 4, max 8)",
#                     "default": 4
#                 }
#             },
#             "required": ["query"]
#         }
#     },
#     {
#         "name": "search_web",
#         "description": (
#             "Search the web for current information. "
#             "Use when the question is about recent events, "
#             "things not in the user's documents, or you need "
#             "to verify a fact."
#         ),
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "A concise web search query"
#                 }
#             },
#             "required": ["query"]
#         }
#     },
#     {
#         "name": "save_note",
#         "description": (
#             "Save a note or summary to a markdown file. "
#             "Use when the user asks to save, record, or "
#             "create a document from the research."
#         ),
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "title": {
#                     "type": "string",
#                     "description": "Filename-safe title for the note"
#                 },
#                 "content": {
#                     "type": "string",
#                     "description": "The full markdown content to save"
#                 }
#             },
#             "required": ["title", "content"]
#         }
#     }
# ]
TOOL_SCHEMAS_OPENAI = [
    {
        "type": "function",           # ← Groq needs this wrapper
        "function": {
            "name": "search_docs",
            "description": "Search the user's uploaded documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results", "default": 4}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note to a markdown file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":   {"type": "string", "description": "Note title"},
                    "content": {"type": "string", "description": "Note content"}
                },
                "required": ["title", "content"]
            }
        }
    }
]


# ─────────────────────────────────────────────
#  Tool executor — maps tool name → real function
# ─────────────────────────────────────────────

class ToolExecutor:
    def __init__(self, rag: RAGRetriever = None):
        self.rag = rag or RAGRetriever()
        os.makedirs("./notes", exist_ok=True)

    def run(self, tool_name: str, tool_args: dict) -> str:
        """
        Dispatch a tool call from the agent.
        Returns a string result that gets sent back to the LLM.
        """
        print(f"\n  [Tool] {tool_name}({json.dumps(tool_args, indent=2)})")

        if tool_name == "search_docs":
            return self._search_docs(**tool_args)

        elif tool_name == "search_web":
            return self._search_web(**tool_args)

        elif tool_name == "save_note":
            return self._save_note(**tool_args)

        else:
            return f"Error: unknown tool '{tool_name}'"

    # ── Tool implementations ──────────────────

    def _search_docs(self, query: str, top_k: int = 4) -> str:
        chunks = self.rag.indexer.search(query, top_k=top_k)

        if not chunks:
            return "No relevant documents found for this query."

        lines = [f"Found {len(chunks)} relevant passages:\n"]
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"[{i}] Source: {c['source']} | Similarity: {c['similarity']}\n"
                f"{c['text'][:400]}\n"
            )
        return "\n".join(lines)

    def _search_web(self, query: str) -> str:
        # Simulated web search for this tutorial.
        # In production, plug in SerpAPI, Tavily, or DuckDuckGo here.
        # e.g.:  results = serpapi.search(query)
        return (
            f"[Simulated web search for: '{query}']\n"
            f"In a real deployment, connect a search API here (Tavily, SerpAPI, etc.).\n"
            f"For now, returning: 'No live web results — using document knowledge only.'"
        )

    def _save_note(self, title: str, content: str) -> str:
        # Make a safe filename from the title
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_"
                             for c in title).strip().replace(" ", "_")
        timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename    = f"./notes/{safe_title}_{timestamp}.md"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"*Saved: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
            f.write(content)

        return f"Note saved successfully to: {filename}"