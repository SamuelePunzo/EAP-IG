# Short-Term Plan: Preserve Precision Without Pretending GQA Is MHA

## Goal

Stabilize the audit suite for grouped-query attention models without weakening the non-GQA guarantees and without claiming per-head precision where the model does not expose it.

## Principles

- Keep strict parity and faithfulness checks for non-GQA families.
- Do not lower thresholds globally to accommodate GQA.
- For GQA, only test quantities that are well-defined at the hook surface the model actually exposes.
- Treat grouped K/V behavior as a first-class architectural constraint, not a special-case nuisance.

## Scope

Applies immediately to:

- `llama`
- `qwen`
- `gemma`
- `mistral` and `ministral` where the bridge hook surface is available

Does not change the public graph semantics yet.

## Plan

### 1. Revert broad test relaxations outside GQA

- Restore strict non-GQA faithfulness expectations.
- Restore strict non-GQA HookedTransformer vs Bridge parity expectations.
- Keep `gpt2` as the main exact-parity and exact-faithfulness reference family.

Reason:
Non-GQA regressions should remain loud.

### 2. Remove ill-posed GQA exact-patching tests

For GQA families, stop using per-query-head exact patching as the gold standard for `k` and `v`.

Reason:
The model exposes shared K/V heads, so a per-query-head exact intervention is not physically realizable for those paths.

### 3. Replace with grouped-equivalence tests

Add GQA-specific tests that validate the correct grouped semantics:

- duplicated `k` scores within the same KV group are equal or near-equal
- duplicated `v` scores within the same KV group are equal or near-equal
- grouped exact patching over a KV group matches grouped EAP-IG attribution
- completeness still holds at the grouped level

Reason:
This preserves a mathematically meaningful test target without inventing fake precision.

### 4. Add grouped aggregation helpers in tests

Introduce test-only helpers that:

- map query heads to KV groups
- aggregate `k/v` scores by KV group
- compare grouped patching effects to grouped attribution scores

Reason:
This isolates the new logic in the audit layer while the core graph remains unchanged.

### 5. Keep bridge certification functional for GQA

Continue running smoke/certification checks for GQA families on:

- load
- prepare
- graph build
- forward pass
- EAP
- EAP-IG
- baseline evaluation
- gradient broadcast sanity

Reason:
Functional support should still be asserted even before exact grouped semantics are represented in the graph.

## Deliverables

- strict non-GQA parity tests
- strict non-GQA exact-faithfulness tests
- new GQA grouped-equivalence tests
- explicit skip/reasoning for ill-posed per-head GQA exact patching
- README note explaining why grouped K/V tests are aggregated

## Acceptance Criteria

- `gpt2` remains the exact reference family and passes strict parity/faithfulness
- GQA families no longer fail because of impossible per-head `k/v` comparisons
- GQA tests still fail on real regressions in grouping, broadcasting, or attribution aggregation
- test output clearly distinguishes:
  - unsupported bridge hook surfaces
  - supported GQA grouped validation
  - supported non-GQA exact validation

## Risks

- Keeping current graph semantics may still invite over-interpretation of per-query-head `k/v` edges.
- Grouped-equivalence checks reduce false failures, but they do not fully solve the representational mismatch.

## Recommended Next Step

Implement grouped GQA audit helpers first, then re-run only:

- GQA smoke/certification tests
- `gpt2` parity/faithfulness reference tests

