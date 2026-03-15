# 5_cli_app.py

from retrieval import retrieve


print("🎓 Speech AI Assistant Ready.")
print("Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ")

    if query.lower() == "exit":
        break

    results = retrieve(query, k=2)

    print("\n--- Relevant Topics ---\n")
    for r in results:
        print(r)
        print("\n" + "-"*60 + "\n")
