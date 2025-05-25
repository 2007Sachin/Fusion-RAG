# Fusion-RAG
 Fusion RAG
Concept : Combines results from multiple retrieval strategies (e.g., BM25 + FAISS) to improve relevance.
Steps :
Retrieve documents using multiple retrievers (e.g., similarity search, MMR).
Merge results using techniques like voting, ranking, or weighted averaging.
Generate an answer using the fused results.
Use Case : Ideal for complex queries where no single retrieval method is optimal.
