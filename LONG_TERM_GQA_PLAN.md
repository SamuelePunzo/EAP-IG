# Long-Term Plan: Make the Graph GQA-Aware

## Goal

Align EAP-IG graph semantics with grouped-query attention architectures so exact patching, parity, and attribution are all defined on the same mathematical object.

## Problem

The current graph assumes that attention backward structure is per query head for all of `q`, `k`, and `v`.

That assumption is valid for standard multi-head attention, but it is false for GQA:

- `q` is per query head
- `k` is per KV head
- `v` is per KV head

When the graph still allocates per-query-head `k/v` edges, it creates distinctions that the underlying model cannot intervene on independently.

## Target Design

Make the graph explicitly asymmetric for GQA families:

- `q` edges indexed by query head
- `k` edges indexed by KV head
- `v` edges indexed by KV head
- MLP and logits unchanged

This should become the canonical representation for grouped-query models.

## Plan

### 1. Introduce GQA-aware graph config

Extend graph config to carry:

- `n_heads`
- `n_key_value_heads`
- `query_to_kv_group` mapping or equivalent derivable metadata
- explicit per-channel dimensions for `q`, `k`, and `v`

Reason:
The graph needs enough structural metadata to size and index each attention pathway honestly.

### 2. Refactor backward indexing

Change graph indexing so the backward slots per layer become:

- `q`: `n_heads`
- `k`: `n_key_value_heads`
- `v`: `n_key_value_heads`
- `mlp`: `1`

instead of the current `3 * n_heads + 1`.

Reason:
This is the core representational correction.

### 3. Update edge construction

Build attention parent edges with:

- one `q` edge per query head
- one `k` edge per KV head
- one `v` edge per KV head

Reason:
Exact patching and score matrices should represent only independently addressable interventions.

### 4. Update attribution code paths

Adjust:

- hook-to-matrix mappings
- gradient accumulation
- intervention construction
- exact patching
- evaluation helpers

so they operate directly on GQA-aware graph indices instead of widening and collapsing duplicate `k/v` edges.

Reason:
The current widen/duplicate approach is a compatibility shim, not a clean long-term abstraction.

### 5. Preserve legacy behavior for non-GQA

Keep the current graph shape and semantics unchanged for:

- `gpt2`
- standard decoder-only MHA families

Reason:
The refactor should not perturb already-correct paths.

### 6. Add migration layer for downstream code

Introduce a thin compatibility layer for any code that currently assumes:

- `3 * n_heads + 1` backward width
- per-query-head indexing for all attention channels

Possible options:

- helper methods that abstract over channel widths
- graph metadata queries instead of raw index arithmetic
- temporary adapters for saved analyses if needed

Reason:
This minimizes downstream breakage while the representation evolves.

### 7. Restore strict exact tests for GQA

Once the graph is GQA-aware, add back strict tests for grouped-query models:

- exact patching faithfulness
- HookedTransformer vs Bridge parity
- grouped completeness
- grouped node/edge attribution consistency

Reason:
At that point the graph and the model expose the same intervention granularity.

## Deliverables

- GQA-aware graph schema
- refactored indexing and edge construction
- updated attribution/evaluation implementation
- restored strict GQA exact tests
- migration notes for downstream analysis code

## Acceptance Criteria

- GQA graph uses `n_key_value_heads` for `k/v` channels
- exact patching is defined without duplicating or collapsing fake per-head `k/v` edges
- strict parity is meaningful again for supported GQA families
- non-GQA behavior remains unchanged
- test code no longer needs grouped-equivalence workarounds for supported GQA models

## Risks

- This is a real internal API change and may affect saved graph assumptions.
- Any code that hardcodes backward matrix widths will need to be updated.
- Some bridge implementations may still lack the necessary hook surface, independent of graph correctness.

## Implementation Order

1. graph config and index refactor
2. attribution/evaluation plumbing
3. test migration
4. re-enable strict GQA exact validation
5. deprecate temporary grouped-equivalence audit helpers

## Recommended Next Step

Before coding, write a short design note that specifies:

- exact backward tensor shapes for MHA vs GQA
- index formulas per channel
- how saved or exported graph data should represent mixed-width attention channels
