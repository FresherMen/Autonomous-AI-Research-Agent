import json
import os
from dotenv import load_dotenv
from groq import Groq
from indexer import DocumentIndexer

load_dotenv()

class ReportGenerator:
    def __init__(self, indexer: DocumentIndexer = None):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not configured in environment or .env file.")
        self.client = Groq(api_key=api_key)
        self.indexer = indexer or DocumentIndexer()

    def bullet_summary(self, topic: str) -> dict:
        chunks = self.indexer.search(topic, top_k=6)
        if not chunks:
            return {"error": "No relevant documents found for this topic.", "raw": ""}

        context = self._format_context(chunks)
        prompt = f"""Based ONLY on the context below, produce a JSON object.
Return ONLY the JSON, no markdown fences, no extra text.

{{
  "title": "A clear title for this summary",
  "key_points": [
    {{"point": "The insight in one sentence", "source": "filename.pdf"}}
  ],
  "one_liner": "A single sentence capturing the core message",
  "confidence": "high or medium or low"
}}

CONTEXT:
{context}

TOPIC: {topic}"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            content = ""
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
            return self._parse_json(content)
        except Exception as e:
            return {"error": f"LLM error: {e}", "raw": ""}

    def research_brief(self, topic: str) -> dict:
        chunks = self.indexer.search(topic, top_k=8)
        if not chunks:
            return {"error": "No relevant documents found for this topic.", "raw": ""}

        context = self._format_context(chunks)
        prompt = f"""Based ONLY on the context below, produce a JSON research brief.
Return ONLY the JSON, no markdown fences, no extra text.

{{
  "title": "Research brief title",
  "abstract": "2-3 sentence overview",
  "sections": [
    {{
      "heading": "Section heading",
      "content": "2-4 sentences of content",
      "sources": ["source1.pdf"]
    }}
  ],
  "key_findings": ["Finding 1", "Finding 2"],
  "gaps": ["What the documents do not cover"],
  "word_count": 0
}}

CONTEXT:
{context}

TOPIC: {topic}"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=2000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            content = ""
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
            result = self._parse_json(content)
            if isinstance(result, dict) and isinstance(result.get("sections"), list):
                total = 0
                for s in result.get('sections', []):
                    if isinstance(s, dict):
                        sec_content = s.get('content') or ""
                        total += len(str(sec_content).split())
                    elif isinstance(s, str):
                        total += len(s.split())
                result['word_count'] = total
            return result
        except Exception as e:
            return {"error": f"LLM error: {e}", "raw": ""}

    def comparison(self, topic_a: str, topic_b: str) -> dict:
        chunks_a = self.indexer.search(topic_a, top_k=4)
        chunks_b = self.indexer.search(topic_b, top_k=4)
        all_chunks = chunks_a + chunks_b
        if not all_chunks:
            return {"error": "No relevant documents found for comparison.", "raw": ""}

        context  = self._format_context(all_chunks)
        prompt = f"""Compare "{topic_a}" and "{topic_b}" based ONLY on the context.
Return ONLY the JSON, no markdown fences, no extra text.

{{
  "topic_a": "{topic_a}",
  "topic_b": "{topic_b}",
  "dimensions": [
    {{
      "aspect": "What is being compared",
      "a": "How topic A relates",
      "b": "How topic B relates",
      "winner": "A or B or tie"
    }}
  ],
  "summary": "One paragraph synthesis",
  "recommendation": "Which to use when and why"
}}

CONTEXT:
{context}"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=1500,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            content = ""
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
            return self._parse_json(content)
        except Exception as e:
            return {"error": f"LLM error: {e}", "raw": ""}

    def render_bullet_summary(self, data: dict) -> str:
        if not isinstance(data, dict):
            return f"Error generating summary: invalid output format ({type(data).__name__})"
        if "error" in data:
            return f"Error generating summary: {data['error']}\n\nRaw: {data.get('raw','')}"
        lines = [f"# {data.get('title', 'Summary')}\n"]
        if data.get('one_liner'):
            lines.append(f"> {data.get('one_liner')}\n")
        lines.append("## Key points\n")
        
        key_points = data.get('key_points', [])
        if isinstance(key_points, str):
            key_points = [{"point": key_points}]
        elif not isinstance(key_points, list):
            key_points = []

        for item in key_points:
            if isinstance(item, dict):
                point = item.get('point', '')
                source = item.get('source', '')
                source_str = f"  *(source: {source})*" if source else ""
                lines.append(f"- {point}{source_str}")
            else:
                lines.append(f"- {str(item)}")
        lines.append(f"\n*Confidence: {data.get('confidence', 'unknown')}*")
        return "\n".join(lines)

    def render_research_brief(self, data: dict) -> str:
        if not isinstance(data, dict):
            return f"Error generating brief: invalid output format ({type(data).__name__})"
        if "error" in data:
            return f"Error generating brief: {data['error']}\n\nRaw: {data.get('raw','')}"
        lines = [f"# {data.get('title', 'Research Brief')}\n"]
        if data.get('abstract'):
            lines.append(f"**Abstract:** {data.get('abstract')}\n")
        
        sections = data.get('sections', [])
        if isinstance(sections, str):
            sections = [{"heading": "Overview", "content": sections}]
        elif not isinstance(sections, list):
            sections = []

        for section in sections:
            if isinstance(section, dict):
                lines.append(f"## {section.get('heading', 'Section')}\n")
                lines.append(str(section.get('content', '')))
                sources = section.get('sources', [])
                if isinstance(sources, list):
                    srcs = ', '.join(str(s) for s in sources)
                else:
                    srcs = str(sources)
                if srcs:
                    lines.append(f"\n*Sources: {srcs}*\n")
            else:
                lines.append(str(section))

        key_findings = data.get('key_findings', [])
        if isinstance(key_findings, str):
            key_findings = [key_findings]
        elif not isinstance(key_findings, list):
            key_findings = []
        if key_findings:
            lines.append("\n## Key findings\n")
            for f in key_findings:
                lines.append(f"- {str(f)}")

        gaps = data.get('gaps', [])
        if isinstance(gaps, str):
            gaps = [gaps]
        elif not isinstance(gaps, list):
            gaps = []
        if gaps:
            lines.append("\n## Gaps in the documents\n")
            for g in gaps:
                lines.append(f"- {str(g)}")

        lines.append(f"\n*~{data.get('word_count', 0)} words*")
        return "\n".join(lines)

    def render_comparison(self, data: dict) -> str:
        if not isinstance(data, dict):
            return f"Error generating comparison: invalid output format ({type(data).__name__})"
        if "error" in data:
            return f"Error generating comparison: {data['error']}\n\nRaw: {data.get('raw','')}"
        a, b = data.get('topic_a', 'A'), data.get('topic_b', 'B')
        lines = [f"# {a} vs {b}\n"]
        lines.append(f"| Aspect | {a} | {b} | Edge |")
        lines.append("|--------|-----|-----|------|")

        dimensions = data.get('dimensions', [])
        if isinstance(dimensions, str):
            dimensions = [{"aspect": "General", "a": dimensions, "b": "-", "winner": "tie"}]
        elif not isinstance(dimensions, list):
            dimensions = []

        for dim in dimensions:
            if isinstance(dim, dict):
                edge = {"A": a, "B": b, "tie": "Tie"}.get(dim.get('winner', 'tie'), 'Tie')
                lines.append(f"| {dim.get('aspect', '')} | {dim.get('a', '')} | {dim.get('b', '')} | {edge} |")
            else:
                lines.append(f"| {str(dim)} | - | - | - |")
        if data.get('summary'):
            lines.append(f"\n**Summary:** {data.get('summary')}")
        if data.get('recommendation'):
            lines.append(f"\n**Recommendation:** {data.get('recommendation')}")
        return "\n".join(lines)

    def _format_context(self, chunks: list) -> str:
        return "\n\n".join(
            f"[{c.get('source', 'unknown')}]\n{c.get('text', '')[:400]}" for c in chunks
        )

    def _parse_json(self, text: str) -> dict:
        clean = text.strip()
        # Strip markdown fences if present
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.lower().startswith("json"):
                    part = part[4:].strip()
                try:
                    res = json.loads(part)
                    if isinstance(res, dict):
                        return res
                except:
                    continue
        try:
            res = json.loads(clean)
            if isinstance(res, dict):
                return res
        except json.JSONDecodeError as e:
            # Try to find JSON object inside the text
            start = clean.find('{')
            end   = clean.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    res = json.loads(clean[start:end+1])
                    if isinstance(res, dict):
                        return res
                except:
                    pass
            return {"error": str(e), "raw": text}
        return {"error": "Failed to parse JSON object from output", "raw": text}
