# Summary: Probe Input Facts (PLE — Step 5)

This synthesis covers all eight fact inputs read from `inputs/` (each file is a
single line of the form `FACT-NN: value-N`, two whitespace-separated fields).

## Fact mapping

- **FACT-01** — value-17
- **FACT-02** — value-34
- **FACT-03** — value-51
- **FACT-04** — value-68
- **FACT-05** — value-85
- **FACT-06** — value-102
- **FACT-07** — value-119
- **FACT-08** — value-136

## Observations

- **Format:** all eight inputs are uniform, one line each (`wc -l` = 1 per
  file, 8 total), and every line has exactly 2 fields.
- **Sequence:** the values form an arithmetic progression on the integer literal
  with first term 17 and a constant common difference of 17
  (17 × n for n = 1 … 8). Extrapolating the pattern, the next term would be
  FACT-09 → value-153.
- **Sums:** the sum of the eight values is 612; the mean is 76.5.
- **Input integrity (verified independently):** `shasum -a 256 inputs/fact-4.txt`
  = `bb06d449f89f647fbf6ee14266ad6f59e5480d8074f075d9236add1e512cc87d`.
- **Duplicates / ordering:** `sort` and `uniq -c` checks on the individual
  inputs confirm each file holds a single unique, canonical entry.

## Outputs produced

- `output/summary.md` — this synthesis.
- `output/matrix.json` — JSON object mapping every FACT-ID to its value.
- `output/receipt.txt` — listing of the three output paths.

## Assumptions

- The `value-N` literal is taken verbatim from each input file; no numeric
  transformation is applied in `matrix.json`.
- Each input file maps to exactly one FACT-ID, keyed by its filename
  (`fact-N.txt` → `FACT-0N`, zero-padded to two digits).