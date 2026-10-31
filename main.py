from retriever import RAGRetriever

def main():
    print("=" * 60)
    print("  RAG Research Assistant — Steps 1+2+3 combined")
    print("=" * 60)
    print("\nCommands:")
    print("  quit     — exit")
    print("  reset    — clear conversation history")
    print("  index    — add a new document (enter path)")
    print("  verbose  — toggle detailed retrieval info")
    print("  sources  — show last answer's source chunks")
    print()

    assistant = RAGRetriever()
    verbose   = False
    last_result = None

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
            assistant.reset_conversation()
            last_result = None
            continue

        if user_input.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
            continue

        if user_input.lower() == "sources":
            if not last_result or not last_result.get('chunks'):
                print("\nNo sources available yet. Ask a question first.\n")
            else:
                print("\nSources from last answer:")
                for c in last_result['chunks']:
                    sim = c.get('similarity')
                    sim_str = f"{sim:.2f}" if isinstance(sim, (int, float)) else "0.00"
                    print(f"  [{sim_str}] {c.get('source', 'unknown')}")
                    print(f"  {c.get('text', '')[:150]}...\n")
            continue

        if user_input.lower() == "index":
            path = input("File path: ").strip()
            count = assistant.indexer.index_file(path)
            print(f"Indexed {count} chunks.")
            continue

        # Normal question — run the RAG pipeline
        last_result = assistant.ask(user_input, verbose=verbose)
        print(f"\nAssistant: {last_result['answer']}\n")

if __name__ == "__main__":
    main()
