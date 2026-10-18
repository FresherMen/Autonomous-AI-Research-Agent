import os
import json
from dotenv import load_dotenv
from groq import Groq
from tools import TOOL_SCHEMAS_OPENAI, ToolExecutor

load_dotenv()

class ResearchAgent:

    MAX_ITERATIONS = 6

    def __init__(self):
        self.client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.executor = ToolExecutor()
        self.history  = []
        self.system = """You are an expert research assistant with access to
three tools: search_docs, search_web, and save_note.
Think step by step, call tools to gather information, then give a clear answer."""

    def run(self, user_goal: str) -> str:
        print(f"\n{'='*60}")
        print(f"Agent goal: {user_goal}")
        print('='*60)

        self.history.append({"role": "user", "content": user_goal})

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n── Iteration {iteration} ─────────────────")

            response = self.client.chat.completions.create(
                model      = "llama-3.3-70b-versatile",
                max_tokens = 2048,
                tools      = TOOL_SCHEMAS_OPENAI,
                messages   = [{"role": "system", "content": self.system}] + self.history
            )

            finish_reason = response.choices[0].finish_reason
            print(f"   Finish reason: {finish_reason}")

            if finish_reason == "tool_calls":
                msg = response.choices[0].message

                # Save assistant message with tool calls to history
                self.history.append({
                    "role":       "assistant",
                    "content":    msg.content or "",
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })

                # Execute each tool and collect results
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)
                    result    = self.executor.run(tool_name, tool_args)
                    print(f"   Result preview: {result[:120]}...")

                    self.history.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result
                    })

            else:
                # end_turn or stop — return final answer
                final_answer = response.choices[0].message.content or ""
                self.history.append({"role": "assistant", "content": final_answer})
                print(f"\n✓ Agent finished in {iteration} iteration(s)")
                return final_answer

        return "Reached maximum steps. Please try a more specific question."

    def reset(self):
        self.history = []
        print("Agent memory cleared.")