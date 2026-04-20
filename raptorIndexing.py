"""
Some questions require very detailed information from a corpus to answer like pertain to a single document or a single chunk, we can call those low level questions.
Some questions require consolidation across broad swast of the document so across many documents or chunks of documents and we can call those like higher level questions.
But we retrieve only k chunks from the vector search and there might be a high level question which may benefit from the other docs which are skipped due to that k filter.

Technique RAPTOR:
It is a way to build a hierarchical index of document summaries.
- We start with a set of documents as our leaves
- We cluster them 
- We summarise each of those clusters
- We do that recursively util we reach some limit or are left with one single cluster (A high level summary of all our documents).
"""