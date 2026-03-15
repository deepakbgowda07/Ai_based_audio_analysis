from retrieval import retrieve

query = "What is the value of joining the AI community?"


results = retrieve(query, k=4)

print("\nTop Relevant Topics:\n")
for r in results:
    print("-" * 60)
    print(r)
    print()
