# Provable Retrieval Bounds for N3 Hyperbolic Memory

**Status:** Documented 2026-11-15, no algorithmic changes made.  
**Scope:** `harness/frontier/n3.py` (SQLite short-term memory, Poincaré 3-ball addressing, FTS5 lexical recall, context management).

---

## 1. Overview & Current Architecture

N3 short-term memory implements a single-database SQLite store (`mem` table and `mem_fts` virtual table) designed to deposit and recall agent observations, decisions, and archived conversation turns.

Every deposited note is assigned:
1. An integer `depth` $\ge 1$ indicating tree hierarchy depth.
2. An address $(x, y, z) \in \mathbb{B}^3$ in the open Poincaré 3-ball ($\|x\|^2 + \|y\|^2 + \|z\|^2 < 1$), where direction is determined by a signed random projection of token hashes and radius is determined by a monotonic radial depth law.
3. An optional parent link `parent` pointing to an earlier memory entry.
4. A boolean flag `pinned` marking core observations that should remain visible across conversation turns.

Recall blends lexical matching (via SQLite FTS5 BM25) with geometric proximity in the Poincaré 3-ball metric.

---

## 2. Step-by-Step Retrieval Algorithm & Complexity Characterization

When `N3.recall(query, k=5, kinds=None)` is called on a database containing $N$ total entries:

### Step 1: Query Tokenization & Normalization
- Extract word tokens from `query`, strip stopwords, and truncate to at most 12 tokens:
  $$T = |\text{tokens}| \le 12$$
- Complexity: $O(|query|)$ string operations.

### Step 2: Candidate Selection
There are two distinct paths for candidate selection:

#### Primary Path: FTS5 BM25 Match Query
When `mem_fts` virtual table is available and $T \ge 1$:
```sql
SELECT m.id, m.kind, m.key, m.text, m.tags, m.x, m.y, m.z, bm25(mem_fts)
FROM mem_fts
JOIN mem m ON m.id = mem_fts.rowid
WHERE mem_fts MATCH ?
ORDER BY bm25(mem_fts)
LIMIT ?
```
with limit parameter `N3_FTS5_CANDIDATE_LIMIT = 40`.

- **Complexity Characterization:**  
  SQLite FTS5 maintains an inverted index (B-tree over term posting lists). Traversing posting lists for $T \le 12$ terms to identify and score matching documents using BM25 and returning the top $C = \min(M, \text{N3\_FTS5\_CANDIDATE\_LIMIT})$ candidates (where $M$ is the number of matching documents in the collection) executes in:
  $$O(C \cdot \log N)$$
  index traversal time.
- **Honest Characterization:**
  - This is **NOT** $O(N)$ full table scanning: the inverted index avoids reading non-matching rows.
  - This is **NOT** $O(\log N)$ nearest-neighbor retrieval: it does not search metric space, but rather extracts up to $C = 40$ lexically matching document IDs.

#### Fallback Path: Bounded Recent Row Scan
When FTS5 is disabled, unavailable (`OperationalError`), or yields 0 hits:
```sql
SELECT id, kind, key, text, tags, x, y, z
FROM mem
ORDER BY id DESC
LIMIT ?
```
with limit parameter `N3_FTS5_FALLBACK_LIMIT = 256`.

- **Complexity Characterization:**  
  The query uses the primary key index (`id DESC`) to read at most $F = 256$ rows. Python filters these $F$ entries against the query tokens and truncates the matching list to at most $C = \text{N3\_FTS5\_CANDIDATE\_LIMIT} = 40$.
- **Complexity:** $O(F) = O(1)$ because $F = 256$ is a fixed compile-time upper bound.
- **Limitation:** Any entry created prior to the last 256 rows is ignored in fallback mode.

### Step 3: Poincaré Query Anchor Placement
- An anchor point is projected for the query using `anchor = place(query, 2)`.
- The direction is computed via BLAKE2b random projection in $O(|query|)$ time.
- The depth is fixed at 2, producing radius $r = \tanh(2 / 2) = \tanh(1) \approx 0.7616$.
- Complexity: $O(1)$ relative to $N$.

### Step 4: Poincaré Distance Metric Evaluation on Candidates
For each candidate entry $r$ in the candidate set ($|candidates| = C \le 40$):
- Compute the exact Poincaré ball metric between `anchor` $u$ and candidate coordinates $v = (r.x, r.y, r.z)$:
  $$d_H(u, v) = 2 \operatorname{asinh}\left(\sqrt{\frac{\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}}\right)$$
- Since $u, v \in \mathbb{R}^3$, the evaluation involves:
  - 3 subtractions and 3 squares for $\|u - v\|^2$,
  - 3 squares for $\|u\|^2$ and $\|v\|^2$,
  - 1 division, 1 square root, and 1 `asinh` function call.
- Each distance calculation executes in strictly $O(1)$ arithmetic operations.
- Combined score:
  $$\text{score} = 0.7 \times \text{lexical\_normalized} + 0.3 \times \min\left(\frac{d_H(u, v)}{8.0}, 1.0\right)$$
- Total metric evaluation for all $C$ candidates:
  $$O(C) \quad \text{where } C \le 40$$

### Step 5: Ranking and Top-$k$ Selection
- Sorting the $C$ scored candidates: $O(C \log C)$.
- Slicing to $k$ entries ($k \le C$): $O(k)$.
- Total post-retrieval time: $O(C \log C) \le 40 \log_2(40) \approx 213$ ops, bounded $O(1)$.

---

## 3. Pinned-View Retrieval & Index Backing

Pinned entries form the core context that is refreshed into the prompt during conversation management (`N3.manage()` calling `N3._refresh_core()`).

### SQL Query & Limit
```sql
SELECT id, kind, key, text
FROM mem
WHERE pinned = 1
ORDER BY id DESC
LIMIT 20
```

### Supporting Index
The schema explicitly defines a compound B-tree index:
```sql
CREATE INDEX IF NOT EXISTS mem_pinned_recent ON mem(pinned, id DESC);
```

### Complexity & Guarantees
- The query performs an exact range lookup on `pinned = 1` and traverses the index in reverse order of `id`.
- Retrieval requires reading at most $K = 20$ index entries and fetching the corresponding table rows.
- Time complexity: $O(K) = O(1)$ index lookups.
- It **never** scans unpinned rows or older pinned rows beyond the 20-entry bound.
- The complete set of pinned notes remains persisted in SQLite (`N3.core(limit=None)` or ID lookup) and does not compact or drop data.

---

## 4. What IS Proven vs. What IS NOT Proven

### What IS Proven
1. **Bounded Candidate Count ($C \le \text{N3\_FTS5\_CANDIDATE\_LIMIT} = 40$):**  
   The candidate set passed to Python for geometric calculation is strictly bounded by $C \le 40$. Neither the database nor the Python runtime ever computes Poincaré distance across the entire corpus of $N$ items during recall.
2. **$O(C)$ Exact Metric Computation:**  
   Poincaré distance calculation is mathematically exact (using numerically stable `asinh`) and strictly bounded to $O(C)$ floating-point operations.
3. **Index-Backed Pinned View:**  
   The core memory prompt view is backed by index `mem_pinned_recent(pinned, id DESC)` and is provably bounded to at most 20 entries ($O(1)$ query time).
4. **Bounded Fallback Scan:**  
   When FTS5 is absent or fails, the row scan is explicitly bounded to `N3_FTS5_FALLBACK_LIMIT = 256` rows ($O(1)$ query time).
5. **Context Memory Preservation:**  
   `N3.manage()` preserves turn structure and tool-call groupings; archived turns remain retrievable by integer row ID via `N3.get(entry_id)`.

### What IS NOT Proven (Explicitly Unproven / Uncharacterized)
1. **Nearest-Neighbor (NN) Completeness:**  
   **UNPROVEN.** The candidate set is selected entirely via BM25 lexical token matching. If an existing note is semantically or geometrically adjacent in Poincaré space but shares no lexical tokens with the query tokens, it has a recall probability of exactly zero ($0\%$). FTS5 does not perform metric space candidate selection.
2. **$O(\log N)$ Total Nearest-Neighbor Retrieval:**  
   **NOT PROVIDED.** There is no hierarchical spatial tree, cover tree, or metric index built on the $(x, y, z)$ coordinates. True nearest-neighbor retrieval in hyperbolic space is not guaranteed in $O(\log N)$ time.
3. **Top-$C$ Truncation Completeness:**  
   If more than $C = 40$ documents match the query terms lexically, SQLite's FTS5 truncates the set based strictly on BM25 rank *before* hyperbolic distances are computed. A document with low BM25 score but near-zero Poincaré distance to the query anchor will be discarded if it falls outside the top 40 lexical hits.
4. **Fallback Completeness:**  
   The 256-row fallback does not inspect records beyond the 256 most recent IDs. Older matching records are omitted entirely.

---

## 5. Radial Placement Law vs. Radial Reuse Policy

### What Exists: Monotonic Radial Placement Law
In `n3.place(text, depth, parent)`:
- Radial distance from the origin $r(d)$ is defined as:
  $$r(d) = \begin{cases}
  \tanh\left(\frac{d}{2}\right) & \text{for } 1 \le d \le 7 \\
  \tanh\left(\frac{7 + \ln(1 + d - 7)}{2}\right) & \text{for } d > 7
  \end{cases}$$
- For all integer depths $d \ge 1$:
  $$0 < r(d) < r(d+1) < 1$$
- For shallow depths ($d \le 7$), the coordinates match legacy behavior.
- For deep descendants ($d > 7$), the logarithmic scaling guarantees strict ordering without numerical clipping to 0.999.
- Child notes blend parent direction: $v_{\text{child}} = \text{normalize}(0.65 \hat{v}_{\text{parent}} + 0.35 v_{\text{own}})$.

### What Does NOT Exist: Radial Reuse & Eviction Policy
- **No Eviction Policy:** There is no mechanism that evicts or archives notes based on radius, depth, or age. Notes remain permanently in SQLite until the database file is removed.
- **No Probability-Based Reuse:** Memory retrieval does not scale sampling probability by radial distance or depth. Depth affects only the geometric coordinate, which contributes a $0.3$ weight to ranking *only if* the note was already captured in the 40-item BM25 candidate set.
- **No Automatic Subtree Pruning:** Deleting or unpinning a parent does not cascade to descendants.

---

## 6. Physical & Thermodynamic Claims (FEP, Energy, Heat)

- **Status:** **None established.**
- **Token Reductions vs. Physical Work:**  
  Conversation carving (moving completed turns into SQLite archives) reduces estimated prompt token counts (e.g., from 4.96M to 0.55M tokens in synthetic simulations). This is an optimization of the API request payload size.
- **Thermodynamic / Free Energy Principle (FEP) Inferences:**  
  No claims regarding thermodynamic entropy, Landauer erasure limits ($k_B T \ln 2$), heat dissipation, physical energy expenditure, or active inference variational free energy are verified, measured, or applicable to this software layer.
- Any inference connecting token budget reductions to physical thermodynamic heat or biological free energy minimization is **unproven and uncharacterized**.

---

## 7. Requirements for Provable $O(\log N)$ Hyperbolic Nearest-Neighbor Retrieval

To upgrade N3 retrieval from a hybrid lexical filter to a provable $O(\log N)$ or $(1+\epsilon)$-approximate nearest-neighbor (ANN) system in hyperbolic space, the following data structures would be required:

1. **Hyperbolic Metric Trees:**
   - **Hyperbolic Cover Trees:** Adapting the cover tree (Beygelzimer, Kakade, Langford 2006) to hyperbolic space $(\mathbb{B}^3, d_H)$. Because the Poincaré metric satisfies the triangle inequality ($d_H(x, z) \le d_H(x, y) + d_H(y, z)$), cover trees provide provable $O(\log N)$ nearest-neighbor search with search time bounded by $O(c^{\Delta} \log N)$ where $c$ is the metric expansion constant.
   - **Vantage-Point Trees (VP-Trees) / Ball Trees:** Partitioning the 3D Poincaré ball using concentric hyperbolic spheres centered at vantage points, pruning spherical shells where $|d_H(p, v) - d_H(q, v)| > r_{\text{current}}$.
2. **Hyperbolic Graph Indices (Hyperbolic HNSW):**
   - Hierarchical Navigable Small World graphs constructed directly in the Poincaré ball or Lorentz hyperboloid model (e.g., matching the formulations of Chami et al. 2019, Sala et al. 2018).
   - Requires hyperbolic distance evaluations during beam search and greedy routing across logarithmic graph layers.
3. **Database Integration:**
   - An SQLite virtual table module (or Rust SQLite extension) indexing Poincaré coordinates via a metric tree, rather than relying on FTS5 inverted text indexes to produce candidate sets.

---

## 8. Summary Table of Bounds

| Component | Mechanism | Proven Bound | Status |
|---|---|---|---|
| FTS5 Candidate Selection | Inverted index posting list scan | $O(C \log N)$, $C \le 40$ | Proven index traversal bound; **No** metric NN completeness |
| No-FTS5 Fallback | Primary key scan on recent rows | $O(F)$, $F \le 256$ | Strictly bounded; misses entries older than 256 rows |
| Poincaré Distance | Closed-form `asinh` in $\mathbb{R}^3$ | $O(1)$ per candidate, $O(C)$ total | Mathematically exact, $O(C)$ verified |
| Pinned View | Compound B-tree (`pinned`, `id DESC`) | $O(K)$, $K \le 20$ | Proven $O(1)$ index lookup |
| Context Carving | Grouped turn archiving in SQLite | Retains boundary invariants | Verified tool-call integrity; no physical heat claims |
| Metric Space NN | Hyperbolic index (Cover tree, HNSW) | $O(\log N)$ NN guarantee | **Not built / Unproven** (requires metric indexing) |

---
*Deliverable for Item 4 of PLE Handoff. Constants defined in `harness/frontier/n3.py` (`N3_FTS5_CANDIDATE_LIMIT`, `N3_FTS5_FALLBACK_LIMIT`). Validated with test suite in `harness/tests/test_frontier.py`.*
