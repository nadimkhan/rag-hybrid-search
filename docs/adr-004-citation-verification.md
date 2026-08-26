# ADR-004: Citation Verification

**Context:** LLMs hallucinate citations — they reference [1] even when [1] doesn't support the claim.

**Decision:** After generation, verify every citation using an LLM-as-judge. Mark verified when score >= 0.7. Flag unverified citations rather than hiding them.

**Consequences:** Extra LLM call per answer. Better signal to users about citation quality. Forces us to track citation_coverage as a first-class metric.
