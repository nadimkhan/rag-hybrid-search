# Generation and Citation System

## Grounded Generation
The generator produces answers using ONLY the retrieved context. The system prompt
instructs the model to cite every factual claim using bracketed references like [1].

## Citation Verification
After generation, every citation is verified using an LLM-as-judge.
A citation is marked verified when the judge's score >= 0.7.
Unverified citations are flagged, not hidden.

## Confidence Scoring
Confidence = 0.5 * retrieval_confidence + 0.5 * citation_coverage
- High: >= 0.85
- Medium: >= 0.60
- Low: < 0.60

## Unanswerable Handling
When retrieval confidence is too low, the system explicitly says it cannot answer
rather than hallucinating.
