# Evaluation Framework

## Golden Dataset
50+ hand-written Q&A pairs covering:
- Simple lookups (easy)
- Multi-hop reasoning (medium)
- Edge cases and failure modes (hard)
- Questions the corpus cannot answer

## Metrics
- **Faithfulness**: Are all claims grounded in retrieved context?
- **Answer correctness**: LLM-as-judge vs. golden answer
- **Retrieval relevance**: Were the right chunks retrieved?
- **Citation accuracy**: Do citations actually support their claims?

## Chunking Strategy Comparison
The eval framework runs the same suite across three chunking strategies:
fixed-size, recursive, and semantic. Results show which strategy wins per document type.
