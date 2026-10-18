from agent import ResearchAgent

def main():
    print("=" * 60)
    print("  Agentic Research Assistant — Step 4")
    print("=" * 60)
    print("This assistant can: search your docs, search the web,")
    print("and save notes to files — all autonomously.\n")
    print("Try asking things like:")
    print("  'What do my papers say about transformers?'")
    print("  'Summarise the key points and save them as a note'")
    print("  'Compare what my docs say vs what's on the web'\n")
    print("Commands: quit, reset\n")

    agent = ResearchAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            continue

        answer = agent.run(user_input)
        print(f"\nAgent: {answer}\n")

if __name__ == "__main__":
    main()