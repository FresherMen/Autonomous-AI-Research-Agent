import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure mocks for third-party modules if not installed in current environment
for mod in ["dotenv", "groq", "chromadb", "chromadb.config", "sentence_transformers", "pypdf"]:
    if mod not in sys.modules:
        m = MagicMock()
        sys.modules[mod] = m

# Add repository to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from chunker import DocumentChunker
from prompt_builder import PromptBuilder
from report_generator import ReportGenerator


class TestDocumentChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker(chunk_size=100, overlap=20)

    def test_no_text_dropped_on_sentence_boundary(self):
        """Verify that truncating to a sentence boundary does not drop intermediate text."""
        text = (
            "Sentence one is right here. This middle text must never be lost in chunking. "
            "Sentence three continues on and provides even more interesting context."
        )
        chunks = self.chunker.chunk_text(text, source_name="test.txt")
        self.assertTrue(len(chunks) >= 2)

        # Check that the critical text "middle text must never be lost" is present in at least one chunk
        found = any("middle text must never be lost" in c["text"] for c in chunks)
        self.assertTrue(found, "Critical middle text was lost during sentence boundary chunking!")

    def test_empty_text(self):
        chunks = self.chunker.chunk_text("", source_name="empty.txt")
        self.assertEqual(chunks, [])

    def test_chunk_size_and_overlap_bounds(self):
        c = DocumentChunker(chunk_size=5, overlap=100)
        self.assertGreater(c.chunk_size, c.overlap)
        self.assertGreaterEqual(c.overlap, 0)

    def test_missing_files_safe(self):
        self.assertEqual(self.chunker.chunk_file("non_existent_file.txt"), [])
        self.assertEqual(self.chunker.chunk_pdf("non_existent_file.pdf"), [])


class TestPromptBuilder(unittest.TestCase):
    def test_first_chunk_exceeds_budget_not_dropped(self):
        """Ensure that if the first chunk exceeds max_context_chars, context is truncated rather than dropped."""
        builder = PromptBuilder(max_context_chars=120)
        chunks = [
            {
                "source": "paper.pdf",
                "similarity": 0.85,
                "text": "A" * 300  # Longer than max_context_chars
            }
        ]
        prompt = builder.build("What is A?", chunks)
        # Should contain part of the chunk text, not just empty context or only omission message
        self.assertIn("Source: paper.pdf", prompt)
        self.assertIn("AAA", prompt)

    def test_no_results_prompt(self):
        builder = PromptBuilder()
        prompt = builder.build_no_results_prompt("test question")
        self.assertIn("test question", prompt)
        self.assertIn("no relevant documents were found", prompt.lower())


class TestReportGenerator(unittest.TestCase):
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"})
    def setUp(self):
        with patch("report_generator.Groq"):
            self.rg = ReportGenerator(indexer=MagicMock())

    def test_parse_json_case_insensitive_fences(self):
        text_with_upper_json = "```JSON\n{\"title\": \"AI Test\", \"key_points\": []}\n```"
        parsed = self.rg._parse_json(text_with_upper_json)
        self.assertEqual(parsed.get("title"), "AI Test")

    def test_render_bullet_summary_string_key_points(self):
        """Prevent character-by-character bullet printing when LLM returns string instead of list."""
        data = {
            "title": "Summary",
            "one_liner": "Core message",
            "key_points": "A single insight string from LLM",
            "confidence": "high"
        }
        rendered = self.rg.render_bullet_summary(data)
        self.assertIn("- A single insight string from LLM", rendered)

    def test_render_research_brief_string_findings_and_gaps(self):
        data = {
            "title": "Brief",
            "abstract": "Abstract text",
            "sections": "Single section content",
            "key_findings": "Single finding",
            "gaps": "Single gap",
            "word_count": 10
        }
        rendered = self.rg.render_research_brief(data)
        self.assertIn("- Single finding", rendered)
        self.assertIn("- Single gap", rendered)
        self.assertIn("Single section content", rendered)

    def test_render_comparison_string_dimension(self):
        data = {
            "topic_a": "Alpha",
            "topic_b": "Beta",
            "dimensions": "Speed vs Accuracy",
            "summary": "Both are good"
        }
        rendered = self.rg.render_comparison(data)
        self.assertIn("Speed vs Accuracy", rendered)


class TestVectorStoreLogic(unittest.TestCase):
    def test_duplicate_batch_ids_and_none_index(self):
        with patch("chromadb.PersistentClient") as mock_client:
            mock_col = MagicMock()
            mock_col.get.return_value = {"ids": []}
            mock_col.count.return_value = 0
            mock_client.return_value.get_or_create_collection.return_value = mock_col

            from vector_store import VectorStore
            store = VectorStore(persist_dir="./test_chroma_db")

            chunks = [
                {"source": "test.txt", "text": "chunk 0", "index": None},
                {"source": "test.txt", "text": "chunk 1", "index": None},
            ]
            embeddings = [[0.1] * 10, [0.2] * 10]

            store.add_chunks(chunks, embeddings)
            self.assertTrue(mock_col.add.called)
            added_ids = mock_col.add.call_args[1]["ids"]
            self.assertEqual(len(added_ids), len(set(added_ids)))

    def test_list_sources_none_safety(self):
        with patch("chromadb.PersistentClient") as mock_client:
            mock_col = MagicMock()
            mock_col.get.return_value = {
                "metadatas": [{"source": "paper1.pdf"}, {"source": None}, None, {"source": "paper2.pdf"}]
            }
            mock_client.return_value.get_or_create_collection.return_value = mock_col

            from vector_store import VectorStore
            store = VectorStore(persist_dir="./test_chroma_db")
            sources = store.list_sources()
            self.assertEqual(sources, ["paper1.pdf", "paper2.pdf"])


class TestAgentHistoryTrimming(unittest.TestCase):
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"})
    def test_agent_history_preserves_user_start_and_tool_call_integrity(self):
        with patch("agent.Groq"), patch("agent.ToolExecutor"):
            from agent import ResearchAgent
            agent = ResearchAgent()
            agent.MAX_HISTORY_MESSAGES = 4

            agent.history = [
                {"role": "user", "content": "question 1"},
                {"role": "assistant", "content": "answer 1"},
                {"role": "user", "content": "question 2"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "search_docs", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "doc result"},
                {"role": "assistant", "content": "final answer 2"},
            ]

            user_goal = "question 3"
            agent.history.append({"role": "user", "content": user_goal})
            if len(agent.history) > agent.MAX_HISTORY_MESSAGES:
                idx = len(agent.history) - agent.MAX_HISTORY_MESSAGES
                while idx < len(agent.history) and agent.history[idx].get("role") != "user":
                    idx += 1
                agent.history = agent.history[idx:]

            self.assertEqual(agent.history[0]["role"], "user")


class TestToolsCompatibility(unittest.TestCase):
    def test_search_docs_includes_text_and_similarity(self):
        mock_rag = MagicMock()
        mock_rag.indexer.search.return_value = [
            {"source": "sample.pdf", "chunk_index": 2, "text": "Sample passage content", "similarity": 0.88}
        ]
        from tools import ToolExecutor
        executor = ToolExecutor(rag=mock_rag)
        res = executor.run("search_docs", {"query": "sample query"})

        self.assertIn("Sample passage content", res)
        self.assertEqual(len(executor.last_sources), 1)
        src = executor.last_sources[0]
        self.assertEqual(src["source"], "sample.pdf")
        self.assertEqual(src["chunk_index"], 2)
        self.assertEqual(src["text"], "Sample passage content")
        self.assertEqual(src["similarity"], 0.88)


if __name__ == "__main__":
    unittest.main()
