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
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "reset":
            assistant.reset_conversation()
            continue

        if user_input.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
            continue

        if user_input.lower() == "sources" and last_result:
            print("\nSources from last answer:")
            for c in last_result['chunks']:
                print(f"  [{c['similarity']:.2f}] {c['source']}")
                print(f"  {c['text'][:150]}...\n")
            continue

        if user_input.lower() == "index":
            path = input("File path: ").strip()
            assistant.indexer.index_file(path)
            continue

        # Normal question — run the RAG pipeline
        last_result = assistant.ask(user_input, verbose=verbose)
        print(f"\nAssistant: {last_result['answer']}\n")

if __name__ == "__main__":
    main()