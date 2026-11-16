# Synthesis of Input Facts

This summary synthesizes all eight fact files found under `inputs/`, each containing exactly one line (8 lines total).

## Facts

- **FACT-01**: value-17 — the baseline fact; `awk '{print NF}'` on this file reports 2 whitespace-separated fields (`FACT-01:` and `value-17`).
- **FACT-02**: value-34 — sorting the file leaves it unchanged (single, already-sorted line).
- **FACT-03**: value-51 — `uniq -c` reports a single occurrence (count 1), confirming no duplicate lines.
- **FACT-04**: value-68 — SHA-256 of the file is `bb06d449f89f647fbf6ee14266ad6f59e5480d8074f075d9236add1e512cc87d`.
- **FACT-05**: value-85.
- **FACT-06**: value-102.
- **FACT-07**: value-119.
- **FACT-08**: value-136.

## Pattern

The values form an arithmetic sequence with a common difference of 17: 17, 34, 51, 68, 85, 102, 119, 136 (i.e., 17 × 1 through 17 × 8).
