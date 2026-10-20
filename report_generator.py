import json
import os
from dotenv import load_dotenv
from groq import Groq
from indexer import DocumentIndexer

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ReportGenerator:
    def __init__(self, indexer: DocumentIndexer = None):
        self.indexer = indexer or DocumentIndexer()

    def bullet_summary(self, topic: str) -> dict:
        chunks = self.indexer.search(topic, top_k=6)
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

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_json(response.choices[0].message.content)

    def research_brief(self, topic: str) -> dict:
        chunks = self.indexer.search(topic, top_k=8)
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

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        result = self._parse_json(response.choices[0].message.content)
        total = sum(len(s['content'].split()) for s in result.get('sections', []))
        result['word_count'] = total
        return result

    def comparison(self, topic_a: str, topic_b: str) -> dict:
        chunks_a = self.indexer.search(topic_a, top_k=4)
        chunks_b = self.indexer.search(topic_b, top_k=4)
        context  = self._format_context(chunks_a + chunks_b)
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

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1500,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_json(response.choices[0].message.content)

    def render_bullet_summary(self, data: dict) -> str:
        if "error" in data:
            return f"Error generating summary: {data['error']}\n\nRaw: {data.get('raw','')}"
        lines = [f"# {data.get('title', 'Summary')}\n"]
        lines.append(f"> {data.get('one_liner', '')}\n")
        lines.append("## Key points\n")
        for item in data.get('key_points', []):
            lines.append(f"- {item['point']}  *(source: {item.get('source','')})*")
        lines.append(f"\n*Confidence: {data.get('confidence', 'unknown')}*")
        return "\n".join(lines)

    def render_research_brief(self, data: dict) -> str:
        if "error" in data:
            return f"Error generating brief: {data['error']}\n\nRaw: {data.get('raw','')}"
        lines = [f"# {data.get('title', 'Research Brief')}\n"]
        lines.append(f"**Abstract:** {data.get('abstract', '')}\n")
        for section in data.get('sections', []):
            lines.append(f"## {section['heading']}\n")
            lines.append(section['content'])
            srcs = ', '.join(section.get('sources', []))
            if srcs:
                lines.append(f"\n*Sources: {srcs}*\n")
        if data.get('key_findings'):
            lines.append("\n## Key findings\n")
            for f in data['key_findings']:
                lines.append(f"- {f}")
        if data.get('gaps'):
            lines.append("\n## Gaps in the documents\n")
            for g in data['gaps']:
                lines.append(f"- {g}")
        lines.append(f"\n*~{data.get('word_count', 0)} words*")
        return "\n".join(lines)

    def render_comparison(self, data: dict) -> str:
        if "error" in data:
            return f"Error generating comparison: {data['error']}\n\nRaw: {data.get('raw','')}"
        a, b = data.get('topic_a', 'A'), data.get('topic_b', 'B')
        lines = [f"# {a} vs {b}\n"]
        lines.append(f"| Aspect | {a} | {b} | Edge |")
        lines.append("|--------|-----|-----|------|")
        for dim in data.get('dimensions', []):
            edge = {"A": a, "B": b, "tie": "Tie"}.get(dim.get('winner', 'tie'), 'Tie')
            lines.append(f"| {dim['aspect']} | {dim['a']} | {dim['b']} | {edge} |")
        lines.append(f"\n**Summary:** {data.get('summary', '')}")
        lines.append(f"\n**Recommendation:** {data.get('recommendation', '')}")
        return "\n".join(lines)

    def _format_context(self, chunks: list) -> str:
        return "\n\n".join(
            f"[{c['source']}]\n{c['text'][:400]}" for c in chunks
        )

    def _parse_json(self, text: str) -> dict:
        clean = text.strip()
        # Strip markdown fences if present
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except:
                    continue
        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            # Try to find JSON object inside the text
            start = clean.find('{')
            end   = clean.rfind('}')
            if start != -1 and end != -1:
                try:
                    return json.loads(clean[start:end+1])
                except:
                    pass
            return {"error": str(e), "raw": text}