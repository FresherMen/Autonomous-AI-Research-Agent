import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class ResearchAssistant:
    """Base Conversational Q&A Engine — Step 1 of 5"""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not configured in environment or .env file.")
        self.client = Groq(api_key=api_key)
        self.history = []
        self.system = (
            "You are an expert research assistant helping users understand "
            "academic papers, technical topics, and complex scientific concepts. "
            "Be clear, concise, and build upon previous context in the conversation."
        )

    def ask(self, question: str) -> str:
        self.history.append({"role": "user", "content": question})

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            temperature=0.3,       # Low temperature for factual precision
            messages=[{"role": "system", "content": self.system}] + self.history
        )

        answer = ""
        if response.choices and response.choices[0].message:
            answer = response.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": answer})

        if hasattr(response, "usage") and response.usage:
            print(f"[tokens used — prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}]")
        return answer

    def reset(self):
        self.history = []
        print("Conversation cleared.")


if __name__ == "__main__":
    bot = ResearchAssistant()
    print("Research Assistant ready. Type 'quit' to exit, 'reset' to clear history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            bot.reset()
            continue

        answer = bot.ask(user_input)
        print(f"\nAssistant: {answer}\n")